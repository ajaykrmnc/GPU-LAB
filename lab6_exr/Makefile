NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

all: build

run: build
	./mat_mul

build: mat_mul

mat_mul: init.o mat_mul.o
	$(NVCC) $^ -o $@

init.o: init.cpp mat_mul.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

mat_mul.o: mat_mul.cu mat_mul.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o
	rm -f mat_mul
