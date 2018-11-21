extern crate nalgebra as na;
extern crate num_traits as num;
extern crate alga as al;

use std::{f64, char, fmt};
use std::ops::{Add, Mul};
use na::{DefaultAllocator, Dim, Real, VectorN};
use na::allocator::Allocator;
use na::dimension::*;

fn main() {
   let x = Dual::<f64, U3>::new(1.0f64);
   println!("{}", x.idx(0) * x.idx(1) * x.idx(1) + 2.0*x.idx(1));
}

//#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<S, D>
where
    S: Real,
    D: Dim,
    DefaultAllocator: Allocator<S, D>,
{
    a: S,
    b: VectorN<S, D>,
}

impl<S, D> Dual<S, D>
where
    S: Real,
    D: Dim + DimName,
    DefaultAllocator: Allocator<S, D>,
{
    pub fn new(value : S) -> Dual<S, D> {
        Dual {
            a: value,
            b: VectorN::<S, D>::from_fn(|_, _| S::one()),
        }
    }
    pub fn idx(&self, index : usize) -> Dual<S, D> {
        Dual {
            a: self.a,
            b: VectorN::<S, D>::from_fn(|i : usize, _| if i == index { S::one() } else { S::zero() } ),
        }
    }

}

impl<S, D> Add for Dual<S, D>
where
    S: Real,
    D: Dim,
    DefaultAllocator: Allocator<S, D>,
{
    type Output = Self;

    fn add(self, rhs: Dual<S, D>) -> Dual<S, D> {
        Dual {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl<S, D> Mul for Dual<S, D>
where
    S: Real + Mul<VectorN<S, D>, Output=VectorN<S, D>>,
    D: Dim,
    DefaultAllocator: Allocator<S, D>,
{
    type Output = Self;

    fn mul(self, rhs: Dual<S, D>) -> Dual<S, D> {
        Dual {
            a: self.a * rhs.a,
            b: self.a * rhs.b + rhs.a * self.b,
        }
    }
}

impl<D> Mul<Dual<f64, D>> for f64
where
    D: Dim,
    DefaultAllocator: Allocator<f64, D>,
{
    type Output = Dual<f64, D>;

    fn mul(self, rhs: Dual<f64, D>) -> Dual<f64, D> {
        Dual {
            a: self * rhs.a,
            b: self * rhs.b,
        }
    }
}
/*impl<S, D> Mul<S> for Dual<S, D>
where
    S: Real + Mul<VectorN<S, D>, Output=VectorN<S, D>>,
    D: Dim,
    DefaultAllocator: Allocator<S, D>,
{
    type Output = Self;

    fn mul(self, rhs: S) -> Dual<S, D> {
        Dual {
            a: self.a * rhs,
            b: self.b * rhs,
        }
    }
}*/

impl<S, D> fmt::Display for Dual<S, D>
where
    S: Real + fmt::Display,
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

impl<S, D> Real for Dual<S, D>
where
    S: Real,
    D: Dim,
    DefaultAllocator: Allocator<S, D>,
{
    fn sin(self) -> Dual<S, D> {
        Dual {
            a : self.a.sin(),
            b : self.b * self.a.cos(),
        }
    }

}
/*impl Real<S, D> for Dual<S, D>
where
    S: Real,
    D: Dim,
    DefaultAllocator: Allocator<S, D>,
{
    fn sin(self) -> Self {
        Dual {
            a : self.a.sin(),
            b : self.b * self.a.cos(),
    }

}*/

/*fn main() {
    let x = lib::Dual::new(2.0f64);
    println!("{}", 100.0f64 + 2.5f64*x*x + 1.0f64);
} */
/* let x = Container<Dual<T>>
 * x[0] should be { x'[0] + [1.0f64 0.0f64 ..]
 *
 * */
/*impl<S, D> Index<usize> for Dual<S, D>
where
    S: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<S, D>,
{
    type Output = Dual<S, D>;

    fn index(&self, index: usize) -> &Dual<S, D> {
        assert!(
            index < self.b.len(),
            "Matrix index out of bounds."
        );
        &Dual {
            a: self.a,
            b: self.b,
        }
    }
}*/
