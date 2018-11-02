extern crate nalgebra as na;
extern crate num_traits as num;

use std::{f64, char, fmt};
use num::One;
use na::{DefaultAllocator, Scalar, Dim, DimName, VectorN};
use na::allocator::Allocator;
use na::dimension::*;

fn main() {
   let k = Dual::<f64, U3>::new(1.0f64);
   println!("{}", k);
}

//#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<S: Scalar, D: Dim>
    where DefaultAllocator: Allocator<S, D>
{
    a: S,
    b: VectorN<S, D>,
}

impl<S, D> Dual<S, D>
where
    S: Scalar + Copy + One,
    D: Dim + DimName,
    DefaultAllocator: Allocator<S, D>,
{
    pub fn new(value : S) -> Dual<S, D> {
        Dual {
            a: value,
            b: VectorN::<S, D>::from_fn(|_, _| S::one() ),
        }
    }
}

impl<S, D> fmt::Display for Dual<S, D>
where
    S: Scalar + fmt::Display,
    D: Dim,
    DefaultAllocator: Allocator<S, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.a);
        for (index, num) in self.b.into_iter().enumerate() {
            write!(f, " + {}\u{03B5}{}", num, char::from_u32(0x2080 + index as u32).unwrap_or('\u{2099}'));
        }
        Ok(())
    }
}

/*fn main() {
    let x = lib::Dual::new(2.0f64);
    println!("{}", 100.0f64 + 2.5f64*x*x + 1.0f64);
} */
/* let x = Container<Dual<T>>
 * x[0] should be { x'[0] + [1.0f64 0.0f64 ..]
 *
 * */
