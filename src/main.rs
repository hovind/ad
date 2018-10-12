mod lib;

fn main() {
    println!("Hello, world!");
    let x = lib::Dual::new(2.0f64);
    println!("{}", 100.0f64 + 2.5f64*x*x + 1.0f64);
}
