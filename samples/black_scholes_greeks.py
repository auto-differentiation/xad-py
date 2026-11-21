##############################################################################
#
#  Sample for option Greeks with first-order adjoint mode in Python
#
#    Prices a European call with the Black-Scholes formula
#      V = S N(d1) - K exp(-r T) N(d2)
#    and computes delta, vega, rho and theta in a single adjoint pass,
#    checked against the closed-form Greeks.
#
#  This file is part of XAD's Python bindings, a comprehensive library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2026 Xcelerit Computing Ltd.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import math

import xad.adj_1st as xadj
from xad import math as xm


def norm_cdf(x):
    # standard normal CDF; works for floats and XAD active types alike
    return 0.5 * (1.0 + xm.erf(x / math.sqrt(2.0)))


def black_scholes_call(spot, strike, rate, vol, maturity):
    sqrt_t = xm.sqrt(maturity)
    d1 = (xm.log(spot / strike) + (rate + 0.5 * vol * vol) * maturity) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return spot * norm_cdf(d1) - strike * xm.exp(-rate * maturity) * norm_cdf(d2)


#  input values
spot = 100.0
strike = 105.0
rate = 0.03  # continuously compounded, per year
vol = 0.20  # per year
maturity = 1.0  # in years

# set independent variables (the strike stays a plain float)
spot_ad = xadj.Real(spot)
vol_ad = xadj.Real(vol)
rate_ad = xadj.Real(rate)
maturity_ad = xadj.Real(maturity)

with xadj.Tape() as tape:
    # and register them
    tape.registerInput(spot_ad)
    tape.registerInput(vol_ad)
    tape.registerInput(rate_ad)
    tape.registerInput(maturity_ad)

    # start recording derivatives
    tape.newRecording()

    # calculate the price
    price = black_scholes_call(spot_ad, strike, rate_ad, vol_ad, maturity_ad)

    # register and seed adjoint of output
    tape.registerOutput(price)
    price.derivative = 1.0

    # compute all Greeks in one backward pass
    tape.computeAdjoints()

    # Theta convention: theta = -dV/dT, the change in value as one year of time
    # passes (maturity shrinks), per year. Divide by 365 for a per-day theta.
    aad = {
        "delta": spot_ad.derivative,
        "vega": vol_ad.derivative,
        "rho": rate_ad.derivative,
        "theta": -maturity_ad.derivative,
    }

# closed-form Black-Scholes Greeks for a European call (no dividends)
sqrt_t = math.sqrt(maturity)
d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * maturity) / (vol * sqrt_t)
d2 = d1 - vol * sqrt_t
pdf_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
discount = math.exp(-rate * maturity)
closed_form = {
    "delta": norm_cdf(d1),
    "vega": spot * pdf_d1 * sqrt_t,
    "rho": strike * maturity * discount * norm_cdf(d2),
    "theta": -spot * pdf_d1 * vol / (2.0 * sqrt_t) - rate * strike * discount * norm_cdf(d2),
}

# output results
print(f"call price = {price.value:.6f}\n")
print(f"{'Greek':8} {'AAD':>14} {'closed form':>14} {'difference':>12}")
for name in aad:
    diff = aad[name] - closed_form[name]
    print(f"{name:8} {aad[name]:14.10f} {closed_form[name]:14.10f} {diff:12.2e}")
print("\ntheta = -dV/dT per year (divide by 365 for per day)")
print("vega and rho are per 1.00 change in vol and rate (divide by 100 for per 1%)")
