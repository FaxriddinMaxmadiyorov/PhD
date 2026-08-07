package main


import "fmt"
import "unicode/utf8"

func main() {
	myString := "❗hello"
	stringLength := len(myString)
	numberOfRunes := utf8.RuneCountInString(myString)

	fmt.Printf("myString - Length: %d - Runes: %d\n", stringLength, numberOfRunes)
}