mod lib;

fn main() {
    println!("Hello, world!");
    let x = lib::Dual::new(2.0f64);
    println!("{}", x*x*x);
}
