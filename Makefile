swingline: swingline.c
	gcc -Wall -Wextra -lglfw -lepoxy -framework OpenGL -g -o $@ $<
clean:
	rm -f swingline
