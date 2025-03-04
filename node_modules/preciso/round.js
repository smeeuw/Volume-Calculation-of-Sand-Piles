"use strict";

const is_imaginary = require("./is_imaginary.js");
const round_rational = require("./round_rational.js");

function round(n, { digits = 0 } = {}) {
  // in case you pass in a numerical string for digits
  digits = Number(digits);

  if (is_imaginary(n)) {
    return round_rational(n.substring(0, n.length - 1), { digits }) + "i";
  } else {
    return round_rational(n, { digits });
  }
}

module.exports = round;
module.exports.default = round;
