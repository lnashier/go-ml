package zson

import "encoding/json"

// Marshal wraps original json.Marshal to hide error
func Marshal(x interface{}) []byte {
	m, _ := json.Marshal(x)
	return m
}

// Unmarshal wraps original json.Unmarshal to hide error
func Unmarshal(data []byte, v interface{}) interface{} {
	_ = json.Unmarshal(data, v)
	return v
}
