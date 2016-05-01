swingline: swingline.c
	gcc -Wall -Wextra -lglfw3 -lepoxy -framework OpenGL -g -o $@ $<
clean:
	rm -f swingline
