---
title: "The Luhn Formula"
tags: [Algorithm, Programming]
date: 2021-11-01
---

The Luhn formula is a simple checksum formula used for validating identification numbers. It is designed to protect against accidental errors in data entry but not malicious attacks. For example, on a website a user may enter a credit card number or ISBN and sometimes these numbers may be mistyped. One can then use the Luhn formula to determine if the entered number is a valid one.

<!--more-->

The validation algorithm is as follows:
1. From the rightmost digit, double the value of every other digit.
2. If a doubled value has two digits then add the digits individually.
3. The number is valid under the Luhn formula: ***if the sum of all the digits is divisible by 10 `(sum mod 10 == 0)`***

For Example, consider the number `1762483`. We want to determine if it is a valid identification number using the Luhn formula?


*Number* | 1 | 7 | 6 | 2 | 4 | 8 | 3
*Double every other digit* |1 | **14** | 6 | **4** | 4 | **16** | 3
*Sum digit* | 1 | **5** | 6 | **4** | 4 | **7** | 3

***Total Sum =*** 1 + 5 + 6 + 4 + 4 + 7 + 3 = 30

The total sum (30) is divisible by 10 and hence `1762483` is valid under the Luhn formula.

Here is the actual implementation in python

```python
def luhn_formula_validator(input):
	'''Check the input number under the Luhn Formula

	Parameters:
	-----------
	input: input number as sting.

	Returns:
	--------
	boolean
	'''

	#convert digits to list
	input_list = [int(x) for x in input]

	position = 1
	sum = 0
	for number in input_list:
		if position % 2 == 0:
			number = number * 2

		#sum individual digits of a doubled digit
		if number > 9:
			number = 1 + number % 10

		sum += number
		position += 1

	return sum % 10 == 0
```
