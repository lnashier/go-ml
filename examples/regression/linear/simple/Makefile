all: clean copy build run

clean:
	@rm -rf $(prb)/scratch

copy:
	@mkdir -p $(prb)/scratch/data/ && cp -R ../../data/$(prb)/ $(prb)/scratch/data/

build:
	@go build -o linear .

run:
	@./linear -prb $(prb)
