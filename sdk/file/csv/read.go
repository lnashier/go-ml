package csv

import (
	"encoding/csv"
	"log"
	"os"

	"github.com/lnashier/go-ml/sdk/data"
)

// Read ...
func Read(schema data.Schema, fpath string) []data.Point {
	f, err := os.Open(fpath)
	if err != nil {
		log.Fatal(err)
	}
	defer func(f *os.File) {
		_ = f.Close()
	}(f)

	// Create a new CSV reader reading from the opened file
	reader := csv.NewReader(f)

	// Read in all the CSV records
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// Read the CSV header
	header := make(map[string]int)
	for j, colName := range records[0] {
		header[colName] = j
	}

	return data.Parse(schema, header, records[1:])
}
