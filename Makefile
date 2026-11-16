# Sur un noeud GH200 de ROMEO :  module load nvhpc/nvhpc-nompi/24.11 && make
all: bin/rtm_gpu bin/rtm_cpu

bin/rtm_gpu: src/rtm_gpu.cu src/rtm.h
	@mkdir -p bin
	nvcc -O3 -std=c++17 -arch=sm_90 -Xptxas -v -o $@ $< 2> bin/ptxas.txt || (cat bin/ptxas.txt; false)

bin/rtm_cpu: src/rtm_cpu.cpp src/rtm.h
	@mkdir -p bin
	nvc++ -O3 -fast -mp -tp=neoverse-v2 -Minfo=vect -o $@ $< 2> bin/minfo_cpu.txt || (cat bin/minfo_cpu.txt; false)

clean:
	rm -rf bin
