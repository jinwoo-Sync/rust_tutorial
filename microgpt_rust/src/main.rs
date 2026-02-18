/// microgpt.rs
/// Rust port of karpathy/microgpt.py (gist 8627fe009c40f57531cb18360106ce95)
///
/// Autograd backward() uses a proper DAG-based topological sort (Kahn's algorithm)
/// via the DiGraph struct below, rather than a DFS stack hack.
///
/// Computation graph edge convention:
///   u -> v  means  v is a consumer of u  (u.prev contains v's children from v's POV)
///   equivalently: edge direction = data-flow direction (input -> output)
///   so loss node has in-degree 0, leaf params have highest in-degree
///   Kahn's BFS then naturally yields: loss first, params last => correct backward order

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;

// ============================================================================
// DiGraph - directed graph with Kahn's topological sort
//
// Nodes are identified by usize (raw pointer cast of Rc<RefCell<ValueInner>>).
// adj[u]  = set of v where edge u -> v exists (u produces v)
// radj[v] = set of u where edge u -> v exists (predecessors of v)
// in-degree(v) = radj[v].len()
// ============================================================================

struct DiGraph {
    adj:  HashMap<usize, HashSet<usize>>,
    radj: HashMap<usize, HashSet<usize>>,
}

impl DiGraph {
    fn new() -> Self {
        DiGraph {
            adj:  HashMap::new(),
            radj: HashMap::new(),
        }
    }

    fn add_node(&mut self, id: usize) {
        self.adj.entry(id).or_default();
        self.radj.entry(id).or_default();
    }

    // edge: u -> v  (u is a dependency of v in the computation graph,
    //                i.e. v was produced from u)
    fn add_edge(&mut self, u: usize, v: usize) {
        self.adj.entry(u).or_default().insert(v);
        self.radj.entry(v).or_default().insert(u);
        self.adj.entry(v).or_default();
        self.radj.entry(u).or_default();
    }

    fn in_degree(&self, v: usize) -> usize {
        self.radj.get(&v).map(|s| s.len()).unwrap_or(0)
    }

    // Kahn's algorithm  O(V + E)
    //
    // Returns nodes in topological order (sources first).
    // In the computation graph this means: output nodes (loss) first,
    // input nodes (params) last => correct order to call backward_fn.
    //
    // If the returned vec length < total node count, a cycle exists.
    // For a valid autograd graph this must never happen.
    fn topological_sort_kahn(&self) -> Vec<usize> {
        let mut in_deg: HashMap<usize, usize> = self
            .adj
            .keys()
            .map(|&v| (v, self.in_degree(v)))
            .collect();

        let mut queue: VecDeque<usize> = in_deg
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&v, _)| v)
            .collect();

        let mut order = Vec::with_capacity(self.adj.len());

        while let Some(u) = queue.pop_front() {
            order.push(u);
            if let Some(succs) = self.adj.get(&u) {
                for &v in succs {
                    let d = in_deg.get_mut(&v).unwrap();
                    *d -= 1;
                    if *d == 0 {
                        queue.push_back(v);
                    }
                }
            }
        }

        order
    }
}

// ============================================================================
// Autograd engine
// ============================================================================

struct ValueInner {
    data:        f64,
    grad:        f64,
    backward_fn: Option<Box<dyn Fn()>>,  // Option으로 take() 가능
    prev:        Vec<Value>,   // input nodes that produced this node
}

#[derive(Clone)]
struct Value(Rc<RefCell<ValueInner>>);

impl Value {
    fn new(data: f64) -> Self {
        Value(Rc::new(RefCell::new(ValueInner {
            data,
            grad: 0.0,
            backward_fn: None,  // leaf 노드는 backward 불필요
            prev: vec![],
        })))
    }

    fn from_op(data: f64, prev: Vec<Value>) -> Self {
        Value(Rc::new(RefCell::new(ValueInner {
            data,
            grad: 0.0,
            backward_fn: None,
            prev,
        })))
    }

    fn data(&self)          -> f64  { self.0.borrow().data }
    fn grad(&self)          -> f64  { self.0.borrow().grad }
    fn add_grad(&self, g: f64)      { self.0.borrow_mut().grad += g; }
    fn set_grad(&self, g: f64)      { self.0.borrow_mut().grad = g; }
    fn set_data(&self, d: f64)      { self.0.borrow_mut().data = d; }
    fn set_fn(&self, f: Box<dyn Fn()>) { self.0.borrow_mut().backward_fn = Some(f); }
    fn id(&self)            -> usize { Rc::as_ptr(&self.0) as usize }

    // ------------------------------------------------------------------
    // backward()
    //
    // Step 1: BFS from self over .prev edges, collect all reachable nodes.
    //
    // Step 2: Build DiGraph.
    //   For each node u and each child c in u.prev:
    //     add edge  c -> u  (c is an input to u, i.e. c produces u)
    //   => in the graph, u has in-degree = number of nodes that use u as input.
    //   => output node (loss) has in-degree 0 (nobody uses loss as input).
    //   => leaf params have in-degree = number of ops that consume them.
    //
    // Step 3: Kahn's topological sort on DiGraph.
    //   Yields: loss, intermediate ops, leaf params.
    //   This is the correct order to call backward_fn.
    //   Every node's backward_fn is called only after all its consumers
    //   have already propagated gradient into it. Chain rule is satisfied.
    // ------------------------------------------------------------------
    fn backward(&self) {
        // Step 1: collect all nodes reachable via .prev
        let mut all: Vec<Value>       = Vec::new();
        let mut seen: HashSet<usize>  = HashSet::new();
        let mut queue: VecDeque<Value> = VecDeque::new();

        queue.push_back(self.clone());
        seen.insert(self.id());

        while let Some(node) = queue.pop_front() {
            let prev = node.0.borrow().prev.clone();
            for child in &prev {
                if seen.insert(child.id()) {
                    queue.push_back(child.clone());
                }
            }
            all.push(node);
        }

        // node id -> Value handle, needed to call backward_fn by id
        let id_map: HashMap<usize, Value> =
            all.iter().map(|n| (n.id(), n.clone())).collect();

        // Step 2: build DiGraph
        let mut graph = DiGraph::new();
        for node in &all {
            graph.add_node(node.id());
        }
        for node in &all {
            // node was produced by its .prev children
            // edge: child -> node  (child is input to node)
            for child in &node.0.borrow().prev {
                graph.add_edge(child.id(), node.id());
            }
        }

        // Step 3: Kahn's topological sort => backward order
        let order = graph.topological_sort_kahn();

        // sanity: must visit all nodes (no cycle in valid autograd graph)
        assert_eq!(order.len(), all.len(), "cycle detected in computation graph");

        self.set_grad(1.0);

        for node_id in order {
            let node = id_map.get(&node_id).unwrap();

            // backward_fn을 take()로 꺼내서 borrow scope 종료
            let backward = {
                let mut inner = node.0.borrow_mut();
                inner.backward_fn.take()
            };
            // 여기서 borrow_mut 해제됨

            // 클로저 실행 중 RefCell 충돌 없음
            if let Some(f) = backward {
                f();
            }
        }
    }
}

// ============================================================================
// Ops
// Closures capture Weak<> to avoid Rc reference cycles.
// ============================================================================

impl std::ops::Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        let out = Value::from_op(self.data() + other.data(), vec![self.clone(), other.clone()]);
        let w = Rc::downgrade(&out.0);
        let (a, b) = (self, other);
        out.set_fn(Box::new(move || {
            if let Some(rc) = w.upgrade() {
                let g = rc.borrow().grad;
                a.add_grad(g);
                b.add_grad(g);
            }
        }));
        out
    }
}

impl std::ops::Add<f64> for Value {
    type Output = Value;
    fn add(self, rhs: f64) -> Value { self + Value::new(rhs) }
}

impl std::ops::Add<Value> for f64 {
    type Output = Value;
    fn add(self, rhs: Value) -> Value { rhs + self }
}

impl std::ops::Mul for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        let out = Value::from_op(self.data() * other.data(), vec![self.clone(), other.clone()]);
        let w = Rc::downgrade(&out.0);
        let (a, b) = (self, other);
        out.set_fn(Box::new(move || {
            if let Some(rc) = w.upgrade() {
                let g = rc.borrow().grad;
                a.add_grad(b.data() * g);
                b.add_grad(a.data() * g);
            }
        }));
        out
    }
}

impl std::ops::Mul<f64> for Value {
    type Output = Value;
    fn mul(self, rhs: f64) -> Value { self * Value::new(rhs) }
}

impl std::ops::Mul<Value> for f64 {
    type Output = Value;
    fn mul(self, rhs: Value) -> Value { rhs * self }
}

impl std::ops::Neg for Value {
    type Output = Value;
    fn neg(self) -> Value { self * -1.0 }
}

impl std::ops::Sub for Value {
    type Output = Value;
    fn sub(self, other: Value) -> Value { self + (-other) }
}

impl std::ops::Sub<f64> for Value {
    type Output = Value;
    fn sub(self, rhs: f64) -> Value { self + (-rhs) }
}

impl std::ops::Div for Value {
    type Output = Value;
    fn div(self, other: Value) -> Value { self * other.powf(-1.0) }
}

impl Value {
    fn powf(self, exp: f64) -> Value {
        let base_data = self.data();
        let out = Value::from_op(base_data.powf(exp), vec![self.clone()]);
        let w = Rc::downgrade(&out.0);
        let base = self;
        out.set_fn(Box::new(move || {
            if let Some(rc) = w.upgrade() {
                let g = rc.borrow().grad;
                base.add_grad(exp * base.data().powf(exp - 1.0) * g);
            }
        }));
        out
    }

    fn log(self) -> Value {
        let out = Value::from_op(self.data().ln(), vec![self.clone()]);
        let w = Rc::downgrade(&out.0);
        let x = self;
        out.set_fn(Box::new(move || {
            if let Some(rc) = w.upgrade() {
                let g = rc.borrow().grad;
                x.add_grad(g / x.data());
            }
        }));
        out
    }

    fn exp(self) -> Value {
        let out = Value::from_op(self.data().exp(), vec![self.clone()]);
        let w = Rc::downgrade(&out.0);
        let x = self;
        out.set_fn(Box::new(move || {
            if let Some(rc) = w.upgrade() {
                let (d, g) = { let b = rc.borrow(); (b.data, b.grad) };
                x.add_grad(d * g);
            }
        }));
        out
    }

    fn relu(self) -> Value {
        let xd = self.data();
        let out = Value::from_op(if xd < 0.0 { 0.0 } else { xd }, vec![self.clone()]);
        let w = Rc::downgrade(&out.0);
        let x = self;
        out.set_fn(Box::new(move || {
            if let Some(rc) = w.upgrade() {
                let (d, g) = { let b = rc.borrow(); (b.data, b.grad) };
                if d > 0.0 { x.add_grad(g); }
            }
        }));
        out
    }
}

// ============================================================================
// Minimal dependency-free PRNG  (xorshift64 + Box-Muller for gauss)
// ============================================================================

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Rng(seed.max(1)) }

    fn u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn f64(&mut self) -> f64 {
        (self.u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn gauss(&mut self, std: f64) -> f64 {
        let u1 = self.f64().max(1e-15);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * std
    }

    fn shuffle<T>(&mut self, v: &mut Vec<T>) {
        for i in (1..v.len()).rev() {
            let j = (self.u64() as usize) % (i + 1);
            v.swap(i, j);
        }
    }

    fn choices(&mut self, w: &[f64]) -> usize {
        let total: f64 = w.iter().sum();
        let mut r = self.f64() * total;
        for (i, &wi) in w.iter().enumerate() {
            r -= wi;
            if r <= 0.0 { return i; }
        }
        w.len() - 1
    }
}

// ============================================================================
// Model ops
// ============================================================================

fn linear(x: &[Value], w: &[Vec<Value>]) -> Vec<Value> {
    w.iter().map(|row| {
        row.iter().zip(x.iter())
            .map(|(wi, xi)| wi.clone() * xi.clone())
            .reduce(|a, b| a + b)
            .unwrap()
    }).collect()
}

fn softmax(logits: Vec<Value>) -> Vec<Value> {
    let max_val = logits.iter().map(|v| v.data()).fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<Value> = logits.into_iter().map(|v| (v - max_val).exp()).collect();
    let total = exps.iter().cloned().reduce(|a, b| a + b).unwrap();
    exps.into_iter().map(|e| e / total.clone()).collect()
}

fn rmsnorm(x: Vec<Value>) -> Vec<Value> {
    let n = x.len() as f64;
    let ms = x.iter()
        .map(|xi| xi.clone() * xi.clone())
        .reduce(|a, b| a + b)
        .unwrap() * (1.0 / n);
    let scale = (ms + 1e-5).powf(-0.5);
    x.into_iter().map(|xi| xi * scale.clone()).collect()
}

// ============================================================================
// GPT model
// ============================================================================

struct Gpt {
    sd:         HashMap<String, Vec<Vec<Value>>>,
    n_layer:    usize,
    n_head:     usize,
    block_size: usize,
    head_dim:   usize,
}

impl Gpt {
    fn new(
        vocab_size: usize,
        n_embd:     usize,
        block_size: usize,
        n_layer:    usize,
        n_head:     usize,
        rng:        &mut Rng,
    ) -> Self {
        let mut mat = |nout: usize, nin: usize, std: f64| -> Vec<Vec<Value>> {
            (0..nout).map(|_| {
                (0..nin).map(|_| Value::new(rng.gauss(std))).collect()
            }).collect()
        };

        let mut sd: HashMap<String, Vec<Vec<Value>>> = HashMap::new();
        sd.insert("wte".into(), mat(vocab_size, n_embd, 0.02));
        sd.insert("wpe".into(), mat(block_size, n_embd, 0.02));

        for i in 0..n_layer {
            sd.insert(format!("l{}.attn_wq", i), mat(n_embd, n_embd, 0.02));
            sd.insert(format!("l{}.attn_wk", i), mat(n_embd, n_embd, 0.02));
            sd.insert(format!("l{}.attn_wv", i), mat(n_embd, n_embd, 0.02));
            sd.insert(format!("l{}.attn_wo", i), mat(n_embd, n_embd, 0.0));
            sd.insert(format!("l{}.mlp_fc1", i), mat(4 * n_embd, n_embd, 0.02));
            sd.insert(format!("l{}.mlp_fc2", i), mat(n_embd, 4 * n_embd, 0.0));
        }

        Gpt { sd, n_layer, n_head, block_size, head_dim: n_embd / n_head }
    }

    fn params(&self) -> Vec<Value> {
        self.sd.values()
            .flat_map(|m| m.iter().flat_map(|r| r.iter().cloned()))
            .collect()
    }

    fn forward(
        &self,
        token_id: usize,
        pos_id:   usize,
        kv_keys:  &mut Vec<Vec<Vec<Value>>>,
        kv_vals:  &mut Vec<Vec<Vec<Value>>>,
    ) -> Vec<Value> {
        let tok = self.sd["wte"][token_id].clone();
        let pos = self.sd["wpe"][pos_id % self.block_size].clone();
        let mut x: Vec<Value> = tok.into_iter().zip(pos).map(|(t, p)| t + p).collect();
        x = rmsnorm(x); // normalize after embedding (matches original)

        for li in 0..self.n_layer {
            // attention
            let residual = x.clone();
            x = rmsnorm(x);
            let q   = linear(&x, &self.sd[&format!("l{}.attn_wq", li)]);
            let k   = linear(&x, &self.sd[&format!("l{}.attn_wk", li)]);
            let val = linear(&x, &self.sd[&format!("l{}.attn_wv", li)]);
            kv_keys[li].push(k);
            kv_vals[li].push(val);

            let scale = (self.head_dim as f64).sqrt();
            let mut x_attn: Vec<Value> = Vec::new();

            for h in 0..self.n_head {
                let hs = h * self.head_dim;
                let he = hs + self.head_dim;
                let q_h = &q[hs..he];

                let attn_logits: Vec<Value> = kv_keys[li].iter().map(|kt| {
                    q_h.iter().zip(&kt[hs..he])
                        .map(|(qi, ki)| qi.clone() * ki.clone())
                        .reduce(|a, b| a + b)
                        .unwrap() * (1.0 / scale)
                }).collect();

                let attn_w = softmax(attn_logits);

                for j in 0..self.head_dim {
                    let out_j = kv_vals[li].iter().zip(attn_w.iter())
                        .map(|(vt, wt)| wt.clone() * vt[hs + j].clone())
                        .reduce(|a, b| a + b)
                        .unwrap();
                    x_attn.push(out_j);
                }
            }

            x = linear(&x_attn, &self.sd[&format!("l{}.attn_wo", li)]);
            x = x.into_iter().zip(residual).map(|(a, b)| a + b).collect();

            // mlp
            let residual = x.clone();
            x = rmsnorm(x);
            x = linear(&x, &self.sd[&format!("l{}.mlp_fc1", li)]);
            x = x.into_iter().map(|xi| xi.relu().powf(2.0)).collect();
            x = linear(&x, &self.sd[&format!("l{}.mlp_fc2", li)]);
            x = x.into_iter().zip(residual).map(|(a, b)| a + b).collect();
        }

        // weight tying: project to vocab with wte^T
        linear(&x, &self.sd["wte"])
    }
}

// ============================================================================
// main
// ============================================================================

fn main() {
    let n_embd:      usize = 16;
    let n_layer:     usize = 1;
    let block_size:  usize = 8;
    let num_steps:   usize = 1000;
    let n_head:      usize = 4;
    let learning_rate: f64 = 1e-2;
    let seed:         u64  = 42;

    let mut rng = Rng::new(seed);

    let text = std::fs::read_to_string("input.txt")
        .expect("input.txt not found - download names.txt from karpathy/makemore and save as input.txt");

    let mut docs: Vec<String> = text.lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    rng.shuffle(&mut docs);

    // character-level tokenizer
    let mut char_set: std::collections::BTreeSet<char> = std::collections::BTreeSet::new();
    for doc in &docs { for ch in doc.chars() { char_set.insert(ch); } }
    let mut chars: Vec<String> = vec!["<BOS>".into(), "<EOS>".into()];
    chars.extend(char_set.into_iter().map(|c| c.to_string()));
    let vocab_size = chars.len();
    let stoi: HashMap<String, usize> = chars.iter().enumerate().map(|(i, c)| (c.clone(), i)).collect();
    let itos: HashMap<usize, String> = chars.iter().enumerate().map(|(i, c)| (i, c.clone())).collect();
    let bos = stoi["<BOS>"];
    let eos = stoi["<EOS>"];
    println!("vocab size: {}, num docs: {}", vocab_size, docs.len());

    let model = Gpt::new(vocab_size, n_embd, block_size, n_layer, n_head, &mut rng);
    let params = model.params();
    println!("num params: {}", params.len());

    // Adam state
    let (beta1, beta2, eps) = (0.9f64, 0.95f64, 1e-8f64);
    let mut m1 = vec![0.0f64; params.len()];
    let mut m2 = vec![0.0f64; params.len()];

    // training loop
    for step in 0..num_steps {
        let doc = &docs[step % docs.len()];
        let tokens: Vec<usize> = std::iter::once(bos)
            .chain(doc.chars().map(|c| stoi[&c.to_string()]))
            .chain(std::iter::once(eos))
            .take(block_size)
            .collect();

        if tokens.len() < 2 { continue; }

        let mut kv_keys: Vec<Vec<Vec<Value>>> = vec![vec![]; n_layer];
        let mut kv_vals: Vec<Vec<Vec<Value>>> = vec![vec![]; n_layer];
        let seq_len = (tokens.len() - 1) as f64;
        let mut lossf = 0.0f64;

        for pos_id in 0..tokens.len() - 1 {
            let logits = model.forward(tokens[pos_id], pos_id, &mut kv_keys, &mut kv_vals);
            let probs  = softmax(logits);
            let loss   = -(probs[tokens[pos_id + 1]].clone().log()) * (1.0 / seq_len);
            lossf += loss.data();
            loss.backward();
        }

        // Adam
        let lr_t = learning_rate * (1.0 - step as f64 / num_steps as f64);
        let t = (step + 1) as f64;
        for (i, p) in params.iter().enumerate() {
            let g = p.grad();
            m1[i] = beta1 * m1[i] + (1.0 - beta1) * g;
            m2[i] = beta2 * m2[i] + (1.0 - beta2) * g * g;
            let m_hat = m1[i] / (1.0 - beta1.powf(t));
            let v_hat = m2[i] / (1.0 - beta2.powf(t));
            p.set_data(p.data() - lr_t * m_hat / (v_hat.sqrt() + eps));
            p.set_grad(0.0);
        }

        println!("step {:4} / {} | loss {:.4}", step + 1, num_steps, lossf);
    }

    // generation
    println!("\n--- generation ---");
    for idx in 0..5 {
        let mut kv_keys: Vec<Vec<Vec<Value>>> = vec![vec![]; n_layer];
        let mut kv_vals: Vec<Vec<Vec<Value>>> = vec![vec![]; n_layer];
        let mut token_id = bos;
        let mut out: Vec<String> = Vec::new();

        for pos_id in 0..block_size {
            let logits  = model.forward(token_id, pos_id, &mut kv_keys, &mut kv_vals);
            let probs   = softmax(logits);
            let weights: Vec<f64> = probs.iter().map(|p| p.data()).collect();
            token_id = rng.choices(&weights);
            if token_id == eos { break; }
            out.push(itos[&token_id].clone());
        }
        println!("sample {}: {}", idx, out.join(""));
    }
}
