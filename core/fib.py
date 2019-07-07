## 


def memoize(f):
    memo = {}
    def helper(x):
        if x not in memo:            
            memo[x] = f(x)
        return memo[x]
    return helper
    
@memoize
def fibonacci_rec(n):
	if n == 0:
		return 0
	elif n == 1:
		return 1
	else:
		val =  fibonacci_rec(n - 1) + fibonacci_rec(n - 2)
		return val


def fibonacci_iter(n):
	_prev, _next =  0, 1
	for _ in range(n):
		yield _prev
		_prev, _next = _next, _prev + _next
print fibonacci_rec(6)
print list(fibonacci_iter(6))
