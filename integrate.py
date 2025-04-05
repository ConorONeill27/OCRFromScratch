# ax^n = (ax^(n+1))/(n+1)

def integrate(a, n):
    return f"({a}x^{n + 1}) / ({n + 1})"

a = 2
n = 3

result = integrate(a, n)
print(result)