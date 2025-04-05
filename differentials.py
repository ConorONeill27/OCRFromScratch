# ax^n = anx^(n-1)

def derivative(a, n):
    if n == 1:
        return f"{a}"
    elif n == 0:
        return "0"
    else:
        return f"{a * n}x^{n - 1}"

a = 35
n = 4

# result = derivative(a, n)
# print(result)


# 6x^2 + 3x - 2
# 12x + 3

print(derivative(6, 2))
print(derivative(3, 1))
print(derivative(-2, 0))