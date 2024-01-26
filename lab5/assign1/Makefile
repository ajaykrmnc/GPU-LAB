NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

all: product

product: init.o Helper.o
	$(NVCC) $^ -o $@

init.o: init.cpp Helper.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

Helper.o: Helper.cu Helper.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o
	rm -f product
