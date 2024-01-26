NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

all: build

run: build
	./sum_reduction

build: sum_reduction

sum_reduction: init.o sum_reduction.o
	$(NVCC) $^ -o $@

init.o: init.cpp sum_reduction.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

sum_reduction.o: sum_reduction.cu sum_reduction.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o
	rm -f sum_reduction
