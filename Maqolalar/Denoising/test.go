package main

import "fmt"

type Greeter interface {
	LanguageName() string
	Greet(name string) string
}

func SayHello(name string, g Greeter) string {
	return fmt.Sprintf(
		"I can speak %s: %s",
		g.LanguageName(),
		g.Greet(name),
	)
}

type GermanGreeter struct{}

func (GermanGreeter) LanguageName() string {
	return "German"
}

func (GermanGreeter) Greet(name string) string {
	return fmt.Sprintf("Hallo %s!", name)
}

func main() {
	germanGreeter := GermanGreeter{}

	fmt.Println(SayHello("Dietrich", germanGreeter))
}