use std::any::type_name;

fn print_type_of<T>(_: &T) {
    println!("{}", type_name::<T>());
}

// Rust의 const는 C++의 constexpr과 비슷합니다.
// 반드시 타입을 명시해야 하며, 컴파일 타임에 값이 정해져야 합니다.
const MAX_POINTS: u32 = 100_000;

fn main() {
    // 1. 기본적으로 Rust의 모든 변수는 '불변(Immutable)'입니다. (C++의 const와 비슷)
    // 아래 x는 값을 변경할 수 없습니다.
    let x: i32 = 5;
    // x = 10; // 만약 이 주석을 풀면 컴파일 에러가 발생합니다!
    println!("Immutable x: {}", x);

    // 2. 값을 변경하고 싶다면 'mut' 키워드를 붙여야 합니다. (Mutable)
    let mut y: i32 = 10;
    println!("Before change y: {}", y);
    y = 20; // mut이 있으므로 변경 가능!
    println!("After change y: {}", y);

    // 3. 상수(const) 사용
    println!("Constant MAX_POINTS: {}", MAX_POINTS);

    /* 요약:
       - let          => 불변 변수 (기본값)
       - let mut      => 가변 변수
       - const        => 상수 (컴파일 타임)
    */
    
    println!("\n--- Previous Type Checks ---");
    let z: bool = true;
    let s: &str = "Hello";
    let string_obj: String = String::from("World");

    print!("z type: "); print_type_of(&z);
    print!("s type: "); print_type_of(&s);
    print!("string_obj type: "); print_type_of(&string_obj);
}
