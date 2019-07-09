LIB=libsolver.so
SRC=solver.c
OBJ=solver.o
OPTIONS=-std=c99 -O3 -Wall -Wextra -pedantic -g
LIBS=-lm -lgsl -lgslcblas
CC=gcc
TEST_BIN=test

default: $(LIB)

$(LIB): $(SRC)
	$(CC) $(OPTIONS) $(SRC) -fPIC -shared -o $(LIB) $(LIBS)



