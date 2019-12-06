#![feature(const_generics)]

extern crate aljabar as al;

use al::{vector, Matrix, Vector, One, Real, Zero, Angle};
use std::cmp::PartialEq;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use std::{char, fmt};

fn f<T>(x: Vector<T, 2>) -> Vector<T, 1>
where
    T: Copy + Debug + Real + Angle + 'static,
{
    vector!((x[0] + x[1] * x[1]).sin())
}

fn main() {
    let x = Dual::jacobian(vector!(1.0f64, 2.0));
    let dx = Dual::hessian(vector!(1.0f64, 2.0));
    let y = Dual::hessian_values(Dual::from(f(dx)));
    println!("{:?}", y.2);
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Dual<T, const N: usize>
where
    /* What is a static type parameter? */
    T: Copy + Debug + Real + 'static,
{
    a: T,
    b: Vector<T, { N }>,
}

impl <T, const N: usize> From<Vector<Dual<T, { N }>, 1>> for Dual<T, { N }>
where
    T: Copy + Debug + Real + 'static,
{
    fn from(u : Vector<Dual<T, { N }>, 1>) -> Self {
        u.x()
    }

}

fn indicator<T>(condition : bool) -> T
where
    T: One + Zero,
{
    if condition { T::one() } else { T::zero() }
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
    T: Copy + Debug + Real + One + Zero,
    Vector<T, { N }>: Copy,
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
    T: Copy + Debug + Real,
    Vector<T, { N }>: Add<Vector<T, { N }>, Output = Vector<T, { N }>> + Copy + Clone,
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
    T: Copy + Debug + Real,
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
    T: Copy + Debug + Real,
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
    T: Copy + Debug + Real,
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
    T: Copy + Debug + Real + Rem<T, Output = T> + Zero,
    Vector<T, { N }>: Copy + Clone + Add<Vector<T, { N }>, Output = Vector<T, { N }>>,
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
    Vector<T, { N }>: Copy,
    T: Copy + Debug + Mul<Vector<T, { N }>, Output = Vector<T, { N }>> + Real,
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
    T: Copy + Debug + Display + PartialOrd + Real + Zero,
    Vector<T, { N }>: Copy,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.a)?;
        for (index, num) in self.b.into_iter().enumerate() {
            /* Do something less awkward here */
            let (sign, num) = if *num < T::zero() { ('-', -(*num)) } else { ('+', *num) };
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
    T: Copy + Debug + Display + PartialOrd + Real,
    Vector<T, { N }>: Copy,
{
    fn partial_cmp(&self, other: &Dual<T, { N }>) -> Option<std::cmp::Ordering> {
        self.a.partial_cmp(&other.a)
    }
}


impl<T, const N: usize> Neg for Dual<T, { N }>
where
    T: Copy + Debug + Real,
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

impl<T, const N: usize> Real for Dual<T, { N }>
where
    T: Copy + Debug + One + Real,
    Vector<T, { N }>: Copy,
{
    fn sqrt(self) -> Self {
        Dual {
            a: self.a.sqrt(),
            b: self.b / ((T::one() + T::one()) * self.a.sqrt()),
        }
    }

    fn mul2(self) -> Self { unimplemented!() }

    fn div2(self) -> Self { unimplemented!() }
}

impl<T, const N: usize> Angle for Dual<T, { N }>
where
    T: Copy + Debug + Angle + One + Real,
    Vector<T, { N }>: Copy,
{

    fn sin(self) -> Self {
        Dual {
            a: self.a.sin(),
            b: self.b * self.a.cos(),
        }
    }
    fn cos(self) -> Self {
        Dual {
            a: self.a.cos(),
            b: -self.b * self.a.sin(),
        }
    }
    fn tan(self) -> Self {
        Dual {
            a: self.a.tan(),
            b: self.b / self.a.cos() / self.a.cos(),
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
}

impl<T, const N: usize> One for Dual<T, { N }>
where
    T : Copy + Debug + Real + One,
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
    T : Copy + Debug + Real + Zero,
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
