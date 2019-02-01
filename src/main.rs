extern crate alga as al;
extern crate nalgebra as na;
extern crate num_traits as num;

use na::allocator::Allocator;
use na::dimension::*;
use na::{DefaultAllocator, Dim, VectorN};
use num::{Float, Num};
use std::cmp::PartialEq;
use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, Rem, Sub, SubAssign};
use std::{char, f64, fmt};

fn f<S>(x: VectorN<S, U2>) -> VectorN<S, U3>
where
    S: Add<S, Output = S>
        + Mul<S, Output = S>
        + Sub<S, Output = S>
        + Debug
        + Copy
        + PartialEq
        + 'static,
{
    VectorN::<S, U3>::new(x[0] + x[1], x[0] * x[1], x[0] - x[1])
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
    /* What is a static type parameter? */
    S: Float + Copy + PartialEq + 'static + Debug,
    D: Dim,
    VectorN<S, D>: Copy,
    DefaultAllocator: Allocator<S, D>,
{
    a: S,
    b: VectorN<S, D>,
}

impl<S, D> Dual<S, D>
where
    S: Copy + Debug + Float,
    D: Dim + DimName,
    VectorN<S, D>: Copy,
    VectorN<Dual<S, D>, D>: Copy,
    DefaultAllocator: Allocator<S, D> + Allocator<Dual<S, D>, D>,
{
    pub fn new(x: VectorN<S, D>) -> VectorN<Dual<S, D>, D> {
        VectorN::<Dual<S, D>, D>::from_fn(|i: usize, _| -> Dual<S, D> {
            Dual {
                a: x[i],
                b: VectorN::<S, D>::from_fn(
                    |j: usize, _| -> S {
                        if i == j {
                            S::one()
                        } else {
                            S::zero()
                        }
                    },
                ),
            }
        })
    }
}

impl<S, D> Add for Dual<S, D>
where
    S: Add<S, Output = S> + Debug + Float,
    D: Dim,
    VectorN<S, D>: Add<VectorN<S, D>, Output = VectorN<S, D>> + Copy + Clone,
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
    S: Debug + Float,
    D: Dim,
    VectorN<S, D>: Sub<VectorN<S, D>, Output = VectorN<S, D>> + Copy + Clone,
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
    S: Debug + Float + Mul<VectorN<S, D>, Output = VectorN<S, D>> + Copy,
    D: Dim,
    VectorN<S, D>: Add<VectorN<S, D>, Output = VectorN<S, D>> + Copy,
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

impl<S, D> Div for Dual<S, D>
where
    S: Debug + Float + Mul<VectorN<S, D>, Output = VectorN<S, D>> + Copy,
    D: Dim,
    VectorN<S, D>:
        Sub<VectorN<S, D>, Output = VectorN<S, D>> + Div<S, Output = VectorN<S, D>> + Copy + Clone,
    DefaultAllocator: Allocator<S, D>,
{
    type Output = Self;
    /* Test this properly, cowboy stuff */
    fn div(self, rhs: Dual<S, D>) -> Dual<S, D> {
        Dual {
            a: self.a / rhs.a,
            b: (self.a * rhs.b - rhs.a * self.b) / self.a / self.a,
        }
    }
}

impl<S, D> Rem for Dual<S, D>
where
    S: Debug + Float + Mul<VectorN<S, D>, Output = VectorN<S, D>> + Copy,
    D: Dim + DimName,
    VectorN<S, D>: Add<VectorN<S, D>, Output = VectorN<S, D>> + Copy + Clone,
    DefaultAllocator: Allocator<S, D>,
{
    type Output = Self;

    fn rem(self, rhs: Dual<S, D>) -> Dual<S, D> {
        let reminder = self.a % rhs.a;
        Dual {
            a: reminder,
            b: if reminder.is_zero() {
                reminder * rhs.b
            } else {
                VectorN::<S, D>::from_fn(|_, _| S::nan())
            },
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

impl<S, D> Display for Dual<S, D>
where
    S: Float + Debug + Display + Copy,
    D: Dim,
    VectorN<S, D>: Copy,
    DefaultAllocator: Allocator<S, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.a)?;
        for (index, num) in self.b.into_iter().enumerate() {
            let sign = if num.is_sign_negative() { '-' } else { '+' };
            write!(
                f,
                " {} {}\u{03B5}{}",
                sign,
                num.abs(),
                char::from_u32(0x2080 + index as u32).unwrap_or('\u{2099}')
            )?;
        }
        Ok(())
    }
}

impl<S, D> std::cmp::PartialOrd for Dual<S, D>
where
    S: Float + Debug + Display + Copy,
    D: Dim,
    VectorN<S, D>: Copy,
    DefaultAllocator: Allocator<S, D>,
{
    fn partial_cmp(&self, other: &Dual<S, D>) -> Option<std::cmp::Ordering> {
        self.a.partial_cmp(&other.a)
    }
}

impl<S, D> num::cast::ToPrimitive for Dual<S, D>
where
    S: Float + Debug + Display + Copy,
    D: Dim,
    VectorN<S, D>: Copy,
    DefaultAllocator: Allocator<S, D>,
{
    fn to_i64(&self) -> Option<i64> {
        self.a.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.a.to_u64()
    }

    fn to_isize(&self) -> Option<isize> {
        self.a.to_isize()
    }

    fn to_i8(&self) -> Option<i8> {
        self.a.to_i8()
    }

    fn to_i16(&self) -> Option<i16> {
        self.a.to_i16()
    }

    fn to_i32(&self) -> Option<i32> {
        self.a.to_i32()
    }

    fn to_i128(&self) -> Option<i128> {
        self.a.to_i128()
    }

    fn to_usize(&self) -> Option<usize> {
        self.a.to_usize()
    }

    fn to_u8(&self) -> Option<u8> {
        self.a.to_u8()
    }

    fn to_u16(&self) -> Option<u16> {
        self.a.to_u16()
    }

    fn to_u32(&self) -> Option<u32> {
        self.a.to_u32()
    }

    fn to_u128(&self) -> Option<u128> {
        self.a.to_u128()
    }

    fn to_f32(&self) -> Option<f32> {
        self.a.to_f32()
    }

    fn to_f64(&self) -> Option<f64> {
        self.a.to_f64()
    }
}

impl<S, D> num::cast::NumCast for Dual<S, D>
where
    S: Float + Debug + Display + Copy,
    D: Dim + DimName,
    VectorN<S, D>: Copy,
    DefaultAllocator: Allocator<S, D>,
{
    fn from<T: num::cast::ToPrimitive>(n: T) -> Option<Self> {
        S::from(n).map(|s: S| -> Dual<S, D> {
            Dual {
                a: s,
                b: VectorN::<S, D>::zeros(),
            }
        })
    }
}

impl<S, D> num::identities::Zero for Dual<S, D>
where
    S: Float + Debug + Display + Copy + AddAssign,
    D: Dim + DimName,
    VectorN<S, D>: Copy,
    DefaultAllocator: Allocator<S, D>,
{
    fn zero() -> Self {
        Dual {
            a: S::zero(),
            b: VectorN::<S, D>::zeros(),
        }
    }
    fn is_zero(&self) -> bool {
        self.a.is_zero()
    }
}

impl<S, D> num::identities::One for Dual<S, D>
where
    S: Debug + Float + Mul<VectorN<S, D>, Output = VectorN<S, D>> + Copy,
    D: Dim + DimName,
    VectorN<S, D>: Add<VectorN<S, D>, Output = VectorN<S, D>> + Copy,
    DefaultAllocator: Allocator<S, D>,
{
    fn one() -> Self {
        Dual {
            a: S::one(),
            b: VectorN::<S, D>::zeros(),
        }
    }
}

impl<S, D, E> Num for Dual<S, D>
where
    S: Num<FromStrRadixErr = E>
        + Float
        + Debug
        + Display
        + Mul<VectorN<S, D>, Output = VectorN<S, D>>
        + Copy
        + AddAssign
        + DivAssign
        + SubAssign,
    D: Dim + DimName,
    VectorN<S, D>: Copy,
    DefaultAllocator: Allocator<S, D>,
{
    type FromStrRadixErr = E;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        S::from_str_radix(str, radix).map(|s: S| -> Dual<S, D> {
            Dual {
                a: s,
                b: VectorN::<S, D>::zeros(),
            }
        })
    }
}

impl<S, D> Neg for Dual<S, D>
where
    S: Float + Debug + Copy,
    D: Dim,
    VectorN<S, D>: Copy,
    DefaultAllocator: Allocator<S, D>,
{
    type Output = Dual<S, D>;

    fn neg(self) -> Self::Output {
        Dual {
            a: -self.a,
            b: -self.b,
        }
    }
}

impl<S, D> Float for Dual<S, D>
where
    S: Num
        + Float
        + Debug
        + Display
        + Mul<VectorN<S, D>, Output = VectorN<S, D>>
        + AddAssign
        + DivAssign
        + SubAssign,
    D: Dim + DimName,
    VectorN<S, D>: Copy,
    DefaultAllocator: Allocator<S, D>,
{
    fn nan() -> Self {
        Dual {
            a: S::nan(),
            b: VectorN::<S, D>::from_fn(|_, _| S::nan()),
        }
    }
    fn infinity() -> Self {
        unimplemented!();
    }
    fn neg_infinity() -> Self {
        unimplemented!();
    }
    fn neg_zero() -> Self {
        unimplemented!();
    }
    fn min_value() -> Self {
        unimplemented!();
    }
    fn min_positive_value() -> Self {
        unimplemented!();
    }
    fn max_value() -> Self {
        unimplemented!();
    }
    fn is_nan(self) -> bool {
        unimplemented!();
    }
    fn is_infinite(self) -> bool {
        unimplemented!();
    }
    fn is_finite(self) -> bool {
        unimplemented!();
    }
    fn is_normal(self) -> bool {
        unimplemented!();
    }
    fn classify(self) -> core::num::FpCategory {
        unimplemented!();
    }
    fn floor(self) -> Self {
        unimplemented!();
    }
    fn ceil(self) -> Self {
        unimplemented!();
    }
    fn round(self) -> Self {
        unimplemented!();
    }
    fn trunc(self) -> Self {
        unimplemented!();
    }
    fn fract(self) -> Self {
        unimplemented!();
    }
    fn abs(self) -> Self {
        unimplemented!();
    }
    fn signum(self) -> Self {
        unimplemented!();
    }
    fn is_sign_positive(self) -> bool {
        unimplemented!();
    }
    fn is_sign_negative(self) -> bool {
        unimplemented!();
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        unimplemented!();
    }
    fn recip(self) -> Self {
        unimplemented!();
    }
    fn powi(self, n: i32) -> Self {
        unimplemented!();
    }
    fn powf(self, n: Self) -> Self {
        unimplemented!();
    }
    fn sqrt(self) -> Self {
        unimplemented!();
    }
    fn exp(self) -> Self {
        unimplemented!();
    }
    fn exp2(self) -> Self {
        unimplemented!();
    }
    fn ln(self) -> Self {
        unimplemented!();
    }
    fn log(self, base: Self) -> Self {
        unimplemented!();
    }
    fn log2(self) -> Self {
        unimplemented!();
    }
    fn log10(self) -> Self {
        unimplemented!();
    }
    fn max(self, other: Self) -> Self {
        unimplemented!();
    }
    fn min(self, other: Self) -> Self {
        unimplemented!();
    }
    fn abs_sub(self, other: Self) -> Self {
        unimplemented!();
    }
    fn cbrt(self) -> Self {
        unimplemented!();
    }
    fn hypot(self, other: Self) -> Self {
        unimplemented!();
    }
    fn sin(self) -> Self {
        unimplemented!();
    }
    fn cos(self) -> Self {
        unimplemented!();
    }
    fn tan(self) -> Self {
        unimplemented!();
    }
    fn asin(self) -> Self {
        unimplemented!();
    }
    fn acos(self) -> Self {
        unimplemented!();
    }
    fn atan(self) -> Self {
        unimplemented!();
    }
    fn atan2(self, other: Self) -> Self {
        unimplemented!();
    }
    fn sin_cos(self) -> (Self, Self) {
        unimplemented!();
    }
    fn exp_m1(self) -> Self {
        unimplemented!();
    }
    fn ln_1p(self) -> Self {
        unimplemented!();
    }
    fn sinh(self) -> Self {
        unimplemented!();
    }
    fn cosh(self) -> Self {
        unimplemented!();
    }
    fn tanh(self) -> Self {
        unimplemented!();
    }
    fn asinh(self) -> Self {
        unimplemented!();
    }
    fn acosh(self) -> Self {
        unimplemented!();
    }
    fn atanh(self) -> Self {
        unimplemented!();
    }
    fn integer_decode(self) -> (u64, i16, i8) {
        unimplemented!();
    }
    fn epsilon() -> Self {
        unimplemented!();
    }
    fn to_degrees(self) -> Self {
        unimplemented!();
    }
    fn to_radians(self) -> Self {
        unimplemented!();
    }
}
