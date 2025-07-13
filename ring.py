def brackets(s: str) -> bool:
    stack = []
    bracket_map = {'(': ')', '{': '}', '[': ']'}
    
    for char in s:
        if char in bracket_map:  # opening bracket
            stack.append(char)
        elif char in bracket_map.values():  # closing bracket
            if not stack or bracket_map[stack.pop()] != char:
                return False
    return not stack

# Test cases
print(brackets("()"))  # True
print(brackets("()[]{}"))  # True
print(brackets("(]"))  # False
print(brackets("({[({[()]})]})"))  # True
print(brackets("{[(and])}"))  # False


num = [2, 4, 22, 44, 65, 78, 4, 34, 11, 22, 33, 44, 55, 66, 77, 77]
target = 98


def Find_index(list,num):
    if num in list:
        print(list.index(num))
    else:
        print("not found num in list")
  
  
Find_index(num,98)  # Output: not found num in list
Find_index(num, 22)  # Output: 2 (first occurrence of 22