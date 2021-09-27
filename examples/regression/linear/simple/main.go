package main

import (
	"flag"
	"fmt"
	"github.com/pkg/errors"
	"log"
	"sync"

	"github.com/spf13/viper"
)

func setup(prb string) (*viper.Viper, error) {
	cfg := viper.New()
	cfg.AddConfigPath(prb)
	cfg.SetConfigName("cfg")
	if err := cfg.ReadInConfig(); err != nil {
		return nil, errors.Wrapf(err, "Failed to load config")
	}
	return cfg, nil
}

func main() {
	fmt.Println("Main sarting...")
	defer fmt.Println("Main done!")

	prb := flag.String("prb", "", "Path to problem dir.")

	flag.Parse()

	cfg, err := setup(*prb)
	if err != nil {
		log.Fatal(err)
	}

	wg := sync.WaitGroup{}

	wg.Add(1)
	go func(cfg *viper.Viper) {
		defer wg.Done()
		if err := (Planner{}).Run(cfg); err != nil {
			fmt.Println(err)
		}
	}(cfg)

	wg.Wait()
}
