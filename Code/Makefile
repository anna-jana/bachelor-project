LIB=libsolver.so
SRC=solver.c

LIB2=libsolver2.so
SRC2=solver2.c

OPTIONS=-std=c99 -O3 -Wall -Wextra -pedantic -g
LIBS=-lm -lgsl -lgslcblas
CC=gcc
TEST_BIN=test

default: $(LIB)

$(LIB): $(SRC)
	$(CC) $(OPTIONS) $(SRC) -fPIC -shared -o $(LIB) $(LIBS)

micro: $(LIB2)

$(LIB2): $(SRC2)
	$(CC) $(OPTIONS) $(SRC2) -fPIC -shared -o $(LIB2) $(LIBS)



