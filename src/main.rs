extern crate nalgebra as na;
extern crate num_traits as num;
extern crate alga as al;

use std::{f64, char, fmt};
use std::fmt::Debug;
use std::cmp::PartialEq;
use std::ops::{Add, Mul, Sub};
use na::{DefaultAllocator, Dim, Real, VectorN};
use na::allocator::Allocator;
use na::dimension::*;

fn f<S>(x: VectorN<S, U2>) -> VectorN<S, U3>
where
    S : Add<S, Output=S>
      + Mul<S, Output=S>
      + Sub<S, Output=S>
      + Debug
      + Copy
      + PartialEq
      + 'static,
{
    VectorN::<S, U3>::new(x[0] + x[1], x[0]*x[1], x[0] - x[1])
}

fn main() {
   let x = VectorN::<f64, U2>::new(1.0f64, 2.0f64);
   let dx = Dual::<f64, U2>::new(x);
   let y = f(dx);
   println!("{}", y);
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Dual<S, D>
where
    S: Real + Copy + PartialEq,
    D: Dim,
    VectorN<S, D>: Copy,
    DefaultAllocator: Allocator<S, D>,
{
    a: S,
    b: VectorN<S, D>,
}

impl<S, D> Dual<S, D>
where
    S: Real + Copy,
    D: Dim + DimName,
    VectorN<S, D>: Copy,
    VectorN<Dual<S, D>, D>: Copy,
    DefaultAllocator: Allocator<S, D> + Allocator<Dual<S, D>, D>,
{
    pub fn new(x : VectorN<S, D>) -> VectorN<Dual<S, D>, D> {
        VectorN::<Dual<S, D>, D>::from_fn(|i : usize, _| -> Dual<S, D> {
            Dual {
                a: x[i],
                b: VectorN::<S, D>::from_fn(|j : usize, _| -> S {
                    if i == j {
                        S::one()
                    } else {
                        S::zero()
                    }
                }),
            }
        })
    }
}

impl<S, D> Add for Dual<S, D>
where
    S: Real,
    D: Dim,
    VectorN<S, D>: Copy + Clone,
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

impl<S, D> Sub for Dual<S, D>
where
    S: Real,
    D: Dim,
    VectorN<S, D>: Copy + Clone,
    DefaultAllocator: Allocator<S, D>,
{
    type Output = Self;

    fn sub(self, rhs: Dual<S, D>) -> Dual<S, D> {
        Dual {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}

impl<S, D> Mul for Dual<S, D>
where
    S: Real + Mul<VectorN<S, D>, Output=VectorN<S, D>> + Copy,
    D: Dim,
    VectorN<S, D>: Copy,
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


/* Should be
 * impl<S, D> Mul<Dual<S, D>> for S
 * where
 *     S: Mul<VectorN<S, D>, Output=VectorN<S, D>> …
 */
impl<D> Mul<Dual<f64, D>> for f64
where
    D: Dim,
    VectorN<f64, D>: Copy,
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

impl<S, D> fmt::Display for Dual<S, D>
where
    S: Real + fmt::Display + Copy,
    D: Dim,
    VectorN<S, D>: Copy,
    DefaultAllocator: Allocator<S, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.a)?;
        for (index, num) in self.b.into_iter().enumerate() {
            let sign = if num.is_sign_negative() { '-' } else { '+' };
            write!(f, " {} {}\u{03B5}{}", sign, num.abs(), char::from_u32(0x2080 + index as u32).unwrap_or('\u{2099}'))?;
        }
        Ok(())
    }
}
