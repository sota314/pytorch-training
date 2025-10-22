def count_words(text,a):
    count = 0
    charList = list(text)
    for char in charList:
        if char == a:
            count += 1
    return count

if __name__ == "__main__":
    s = "hello world"
    print("e:",count_words(s,"e"))
    print("l:",count_words(s,"l"))
    print("o:",count_words(s,"o"))