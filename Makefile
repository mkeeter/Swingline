swingline: swingline.c
	gcc -Wall -Wextra -lglfw3 -lepoxy -lpng -framework OpenGL -g -o $@ $<
clean:
	rm -f swingline
