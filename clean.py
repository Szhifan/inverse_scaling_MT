
strings = "huuuh"
def longest_anagram(s:list):
    n = len(s)
    dp = [[0]*n for _ in range(n)]
    max_len = 0 
    for i in range(n):
        dp[i][i] = 1 
    for l in range(2,n+1):
        for i in range(n):
            j = i + l - 1
            if j >= n:
                break 
            if s[i] == s[j]:
                if l == 2:
                    dp[i][j] = 2 
                    max_len = max(max_len,l)
                else:
                    dp[i][j] = dp[i+1][j-1]
                    max_len = max(max_len,l)
    return max_len
print(longest_anagram(strings))
                