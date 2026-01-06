from utils.word_counter import count_words

if __name__ == "__main__":
    s = "hello world"
    print("e:",count_words(s,"e"))
    print("l:",count_words(s,"l"))
    print("o:",count_words(s,"o"))