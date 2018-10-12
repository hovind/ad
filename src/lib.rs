use std::f64;
use std::fmt;
use std::ops::Add;
use std::ops::Mul;


#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    a: f64,
    b: f64,
}

impl fmt::Display for Dual {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}\u{03B5}", self.a, self.b)
    }
}

impl Dual {
    pub fn new(value: f64) -> Dual {
        Dual {
            a: value,
            b: 1.0f64
        }
    }
}

impl Add for Dual {
    type Output = Self;

    fn add(self, rhs: Dual) -> Dual {
        Dual {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl Mul for Dual {
    type Output = Self;

    fn mul(self, rhs: Dual) -> Dual {
        Dual {
            a: self.a + rhs.a,
            b: self.a * rhs.b + rhs.a * self.b,
        }
    }
}

