#![feature(const_generics)]


#[cfg(test)]
mod tests {
    use super::*;
    use al::{vector, matrix};
    #[test]
    fn hessian() {
        fn f<T>(x: Vector<T, 2>) -> Vector<T, 1>
        where
            T: 'static + Copy + Debug + Float,
        {
            vector!((x[0] + x[1] * x[1]))
        }

        let x = Dual::hessian(vector!(1.0f64, 2.0));
        let y = Dual::hessian_values(Dual::from(f(x)));
        let value = -0.9589242746631385f64;
        let jacobian = vector![ 0.28366218546322625f64, 1.134648741852905 ]; 
        let hessian = matrix![[ 0.9589242746631385f64, 3.835697098652554 ],
                              [ 3.835697098652554, 15.910112765536669 ]];
        assert_eq!(y.0, value);
        assert_eq!(y.1, jacobian);
        assert_eq!(y.2, hessian);

    }
}

extern crate aljabar as al;
extern crate num_traits as num;


use al::{Matrix, Vector};
use num::{
    Float,
    identities::{One,Zero}
};
use std::{
    char,
    fmt,
    cmp::PartialEq,
    fmt::{Debug, Display},
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Dual<T, const N: usize>
{
    a: T,
    b: Vector<T, { N }>,
}

impl <T, const N: usize> From<Vector<Dual<T, { N }>, 1>> for Dual<T, { N }>
where
    T: Copy + Debug + 'static,
{
    fn from(u : Vector<Dual<T, { N }>, 1>) -> Self {
        u.x()
    }

}

fn unit<T, const N: usize>(i : usize) -> Vector<T, { N }>
where
    T : One + Zero,
{
    let mut u = Vector::<T, { N }>::zero();
    u[i] = T::one();
    u
}

impl<T, const N: usize> Dual<T, { N }>
where
    T: Copy + Debug + PartialEq + One + Zero,
    Vector<T, { N }>: Copy + Zero,
    Vector<Dual<T, { N }>, { N }>: Copy,
{
    pub fn jacobian(v: Vector<T, { N }>) -> Vector<Dual<T, { N }>, { N }> {
        v.indexed_map(|i: usize, x: T| -> Dual<T, { N }> {
            Dual {
                a: x,
                b: unit(i),
            }
        })
    }
    pub fn hessian(v: Vector<T, { N }>) -> Vector<Dual<Dual<T, { N }>, { N }>, { N }> {
        v.indexed_map(|i: usize, x: T| -> Dual<Dual<T, { N }>, { N }> {
            Dual {
                a: Dual {
                    a: x,
                    b: unit(i),
                },
                b: unit::<Dual<T, { N }>, N>(i),
            }
        })
    }

    pub fn hessian_values(v : Dual<Dual<T, { N }>, { N }>) -> (T, Vector<T, { N }>, Matrix<T, { N }, { N }>) {
        let rows : [[T; { N }]; { N }] = Vector::into(v.b.map(|u : Dual<T, { N }>| -> [T; { N }] { Vector::into(u.b) }));
        (v.a.a,
         v.a.b,
         Matrix::from(rows),
        )
    }
}

impl<T, const N: usize> Add for Dual<T, { N }>
where
    T: Add<T, Output = T> + Copy + Debug,
    Vector<T, { N }>: Add<Vector<T, { N }>, Output = Vector<T, { N }>>,
{
    type Output = Self;

    fn add(self, rhs: Dual<T, { N }>) -> Dual<T, { N }> {
        Dual {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}
impl<T, const N: usize> Sub for Dual<T, { N }>
where
    T: Sub<T, Output = T> + Copy + Debug,
    Vector<T, { N }>: Sub<Vector<T, { N }>, Output = Vector<T, { N }>> + Copy + Clone,
{
    type Output = Self;

    fn sub(self, rhs: Dual<T, { N }>) -> Dual<T, { N }> {
        Dual {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}

impl<T, const N: usize> Mul for Dual<T, { N }>
where
    T: Mul<T, Output = T> + Copy + Debug,
    Vector<T, { N }>: Add<Vector<T, { N }>, Output = Vector<T, { N }>> + Copy,
{
    type Output = Self;

    fn mul(self, rhs: Dual<T, { N }>) -> Dual<T, { N }> {
        Dual {
            a: self.a * rhs.a,
            b: rhs.b * self.a + self.b * rhs.a,
        }
    }
}

impl<T, const N: usize> Div for Dual<T, { N }>
where
    T: Div<T, Output = T> + Copy + Debug,
    Vector<T, { N }>: Copy + Clone
                    + Sub<Vector<T, { N }>, Output = Vector<T, { N }>>
                    + Div<T, Output = Vector<T, { N }>>
                    + Mul<T, Output = Vector<T, { N }>>,
{
    type Output = Self;
    /* Test this properly, cowboy stuff */
    fn div(self, rhs: Dual<T, { N }>) -> Dual<T, { N }> {
        Dual {
            a: self.a / rhs.a,
            b: (rhs.b  * self.a - self.b * rhs.a) / self.a / self.a,
        }
    }
}

impl<T, const N: usize> Rem for Dual<T, { N }>
where
    T: Copy + Debug + Rem<T, Output = T> + Zero,
    Vector<T, { N }>: Mul<T, Output = Vector<T, { N }>>,
{
    type Output = Self;

    fn rem(self, rhs: Dual<T, { N }>) -> Dual<T, { N }> {
        let reminder = self.a % rhs.a;
        Dual {
            a: reminder,
            b: rhs.b * reminder,
        }
    }
}

impl<T, const N: usize> Mul<T> for Dual<T, { N }>
where
    T: Copy + Debug + Mul<T, Output = T>,
    Vector<T, { N }>: Mul<T, Output = Vector<T, { N }>>,
{
    type Output = Dual<T, { N }>;

    fn mul(self, rhs: T) -> Dual<T, { N }> {
        Dual {
            a: self.a * rhs,
            b: self.b * rhs,
        }
    }
}

impl<T, const N: usize> Display for Dual<T, { N }>
where
    T: Copy + Debug + Display + Neg<Output = T> + PartialOrd + Zero,
    Vector<T, { N }>: Copy,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.a)?;
        for (index, &num) in self.b.into_iter().enumerate() {
            /* Do something less awkward here */
            let (sign, num) = if num < T::zero() { ('-', -num) } else { ('+', num) };
            write!(
                f,
                " {} {}\u{03B5}{}",
                sign,
                num,
                char::from_u32(0x2080 + index as u32).unwrap_or('\u{2099}')
            )?;
        }
        Ok(())
    }
}

impl<T, const N: usize> std::cmp::PartialOrd for Dual<T, { N }>
where
    T: Copy + Debug + Display + PartialOrd,
    Vector<T, { N }>: Copy,
{
    fn partial_cmp(&self, other: &Dual<T, { N }>) -> Option<std::cmp::Ordering> {
        self.a.partial_cmp(&other.a)
    }
}


impl<T, const N: usize> Neg for Dual<T, { N }>
where
    T: Copy + Debug + Neg<Output = T>,
    Vector<T, { N }>: Copy,
{
    type Output = Dual<T, { N }>;

    fn neg(self) -> Self::Output {
        Dual {
            a: -self.a,
            b: -self.b,
        }
    }
}

impl<T, const N: usize> One for Dual<T, { N }>
where
    T : Copy + Debug + One + PartialEq,
    Vector<T, { N }> : Zero,
{
    fn one() -> Self {
        Dual {
            a: T::one(),
            b: Vector::<T, { N }>::zero(),
        }
    }

    fn is_one(&self) -> bool {
        self.a.is_one()
    }
}

impl<T, const N: usize> Zero for Dual<T, { N }>
where
    T : Copy + Debug + Zero,
{
    fn zero() -> Self {
        Dual {
            a: T::zero(),
            b: Vector::<T, { N }>::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.a.is_zero()
    }
}

impl<T, const N: usize> num::cast::ToPrimitive for Dual<T, { N }>
where
    T: num::cast::ToPrimitive,
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

impl<T, const N: usize> num::cast::NumCast for Dual<T, { N }>
where
    T: num::cast::NumCast + Zero,
{
    fn from<S: num::cast::ToPrimitive>(n: S) -> Option<Self> {
        T::from(n).map(|t : T| -> Self {
            Dual {
                a: t,
                b: Vector::<T, { N }>::zero(),
            }
        })
    }
}

impl<T, E, const N: usize> num::Num for Dual<T, { N }> where
    T: Copy + Debug + num::Num<FromStrRadixErr = E>
{
    type FromStrRadixErr = E;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(str, radix).map(|t : T| -> Self {
            Dual {
                a: t,
                b: Vector::<T, { N }>::zero(),
            }
        })
    }
}

impl<T, const N: usize> Float for Dual<T, { N }> where
    T: Copy + Debug + Display + Float,
{
    fn nan() -> Self {
        Dual {
            a: T::nan(),
            b: Vector::<T, { N }>::zero(),
        }
    }
    fn infinity() -> Self {
        Dual {
            a: T::infinity(),
            b: Vector::<T, { N }>::zero(),
        }
    }
    fn neg_infinity() -> Self {
        Dual {
            a: T::neg_infinity(),
            b: Vector::<T, { N }>::zero(),
        }
    }
    fn neg_zero() -> Self {
        Dual {
            a: T::neg_zero(),
            b: Vector::<T, { N }>::zero(),
        }
    }
    fn min_value() -> Self {
        Dual {
            a: T::min_value(),
            b: Vector::<T, { N }>::zero(),
        }
    }
    fn min_positive_value() -> Self {
        Dual {
            a: T::min_positive_value(),
            b: Vector::<T, { N }>::zero(),
        }
    }
    fn max_value() -> Self {
        Dual {
            a: T::max_value(),
            b: Vector::<T, { N }>::zero(),
        }
    }
    fn is_nan(self) -> bool {
        self.a.is_nan()
    }
    fn is_infinite(self) -> bool {
        self.a.is_infinite()
    }
    fn is_finite(self) -> bool {
        self.a.is_finite()
    }
    fn is_normal(self) -> bool {
        self.a.is_normal()
    }
    fn classify(self) -> core::num::FpCategory {
        self.a.classify()
    }
    fn floor(self) -> Self {
        Dual {
            a: self.a.floor(),
            b: Vector::<T, { N }>::from_fn(|_| T::nan()),
        }
    }
    fn ceil(self) -> Self {
        Dual {
            a: self.a.ceil(),
            b: Vector::<T, { N }>::from_fn(|_| T::nan()),
        }
    }
    fn round(self) -> Self {
        Dual {
            a: self.a.round(),
            b: Vector::<T, { N }>::from_fn(|_| T::nan()),
        }
    }
    fn trunc(self) -> Self {
        Dual {
            a: self.a.trunc(),
            b: Vector::<T, { N }>::from_fn(|_| T::nan()),
        }
    }
    fn fract(self) -> Self {
        Dual {
            a: self.a.fract(),
            b: Vector::<T, { N }>::from_fn(|_| T::nan()),
        }
    }
    fn abs(self) -> Self {
        Dual {
            a: self.a.abs(),
            b: if self.a.is_sign_positive() {
                self.b
            } else if self.a.is_sign_negative() {
                -self.b
            } else {
                Vector::<T, { N }>::from_fn(|_| T::nan())
            },
        }
    }
    fn signum(self) -> Self {
        Dual {
            a: self.a.signum(),
            b: Vector::<T, { N }>::from_fn(|_| {
                if self.a.is_zero() {
                    T::nan()
                } else {
                    T::zero()
                }
            }),
        }
    }
    fn is_sign_positive(self) -> bool {
        self.a.is_sign_positive()
    }
    fn is_sign_negative(self) -> bool {
        self.a.is_sign_negative()
    }
    fn mul_add(self, _a: Self, _b: Self) -> Self {
        unimplemented!();
    }
    fn recip(self) -> Self {
        Dual {
            a: self.a.recip(),
            b: -self.b / self.a / self.a,
        }
    }
    fn powi(self, n: i32) -> Self {
        Dual {
            a: self.a.powi(n),
            b: T::from(n).unwrap_or(T::nan()) * self.a.powi(n - 1) * self.b,
        }
    }
    fn powf(self, _n: Self) -> Self {
        unimplemented!();
    }
    fn sqrt(self) -> Self {
        Dual {
            a: self.a.sqrt(),
            b: self.b / ((T::one() + T::one()) * self.a.sqrt()),
        }
    }
    fn exp(self) -> Self {
        Dual {
            a: self.a.exp(),
            b: self.a.exp() * self.b,
        }
    }
    fn exp2(self) -> Self {
        unimplemented!();
    }
    fn ln(self) -> Self {
        Dual {
            a: self.a.ln(),
            b: self.b / self.a,
        }
    }
    fn log(self, _base: Self) -> Self {
        unimplemented!();
    }
    fn log2(self) -> Self {
        unimplemented!();
    }
    fn log10(self) -> Self {
        unimplemented!();
    }
    fn max(self, _other: Self) -> Self {
        unimplemented!();
    }
    fn min(self, _other: Self) -> Self {
        unimplemented!();
    }
    fn abs_sub(self, other: Self) -> Self {
        (self - other).abs()
    }
    fn cbrt(self) -> Self {
        Dual {
            a: self.a.cbrt(),
            b: self.b / ((T::one() + T::one() + T::one()) * self.a.cbrt() * self.a.cbrt()),
        }
    }
    fn hypot(self, _other: Self) -> Self {
        unimplemented!();
    }
    fn sin(self) -> Self {
        Dual {
            a: self.a.sin(),
            b: self.a.cos() * self.b,
        }
    }
    fn cos(self) -> Self {
        Dual {
            a: self.a.cos(),
            b: -self.a.sin() * self.b,
        }
    }
    fn tan(self) -> Self {
        Dual {
            a: self.a.tan(),
            b: self.b / self.a.cos() / self.a.cos(),
        }
    }
    fn asin(self) -> Self {
        Dual {
            a: self.a.asin(),
            b: self.b / (T::one() - self.a * self.a).sqrt(),
        }
    }
    fn acos(self) -> Self {
        Dual {
            a: self.a.asin(),
            b: -self.b / (T::one() - self.a * self.a).sqrt(),
        }
    }
    fn atan(self) -> Self {
        Dual {
            a: self.a.atan(),
            b: self.b / (T::one() + self.a * self.a),
        }
    }
    fn atan2(self, other: Self) -> Self {
        let squared_distance = self.a * self.a + other.a * other.a;
        Dual {
            a: self.a.atan2(other.a),
            b: -self.b * other.a / squared_distance + other.b * self.a / squared_distance,
        }
    }
    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.a.sin_cos();
        (
            Dual {
                a: sin,
                b: self.b * cos,
            },
            Dual {
                a: cos,
                b: -self.b * sin,
            },
        )
    }
    fn exp_m1(self) -> Self {
        Dual {
            a: self.a.exp_m1(),
            b: self.a.exp() * self.b,
        }
    }
    fn ln_1p(self) -> Self {
        Dual {
            a: self.a.ln_1p(),
            b: self.b / (self.a + T::one()),
        }
    }
    fn sinh(self) -> Self {
        Dual {
            a: self.a.sinh(),
            b: self.a.cosh() * self.b,
        }
    }
    fn cosh(self) -> Self {
        Dual {
            a: self.a.cosh(),
            b: self.a.sinh() * self.b,
        }
    }
    fn tanh(self) -> Self {
        Dual {
            a: self.a.tanh(),
            b: (T::one() - self.a.tanh() * self.a.tanh()) * self.b,
        }
    }
    fn asinh(self) -> Self {
        Dual {
            a: self.a.asinh(),
            b: self.b / (self.a * self.a + T::one()).sqrt(),
        }
    }
    fn acosh(self) -> Self {
        Dual {
            a: self.a.acosh(),
            b: self.b / (self.a * self.a - T::one()).sqrt(),
        }
    }
    fn atanh(self) -> Self {
        Dual {
            a: self.a.atanh(),
            b: self.b / (T::one() - self.a * self.a),
        }
    }
    fn integer_decode(self) -> (u64, i16, i8) {
        self.a.integer_decode()
    }
    fn epsilon() -> Self {
        Dual {
            a: T::epsilon(),
            b: Vector::<T, { N }>::zero(),
        }
    }
    fn to_degrees(self) -> Self {
        unimplemented!();
    }
    fn to_radians(self) -> Self {
        unimplemented!();
    }
}
