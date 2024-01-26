NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

all: addMul

addMul: init.o AddMul.o
	$(NVCC) $^ -o $@

init.o: init.cpp AddMul.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

AddMul.o: AddMul.cu AddMul.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o
	rm -f addMul

