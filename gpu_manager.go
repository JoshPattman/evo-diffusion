package main

import "github.com/rajveermalviya/go-webgpu/wgpu"

type Manager struct {
	Instance *wgpu.Instance
	Adapter  *wgpu.Adapter
	Device   *wgpu.Device
}

func NewManager() (*Manager, error) {
	succsessful := false
	m := &Manager{}
	var err error
	defer func() {
		if !succsessful {
			m.Release()
		}
	}()

	// Get instance
	m.Instance = wgpu.CreateInstance(nil)
	// Get adapter
	m.Adapter, err = m.Instance.RequestAdapter(&wgpu.RequestAdapterOptions{})
	if err != nil {
		return nil, err
	}
	// Get device
	m.Device, err = m.Adapter.RequestDevice(nil)
	if err != nil {
		return nil, err
	}
	succsessful = true
	return m, nil
}

func (m *Manager) Release() {
	if m.Device != nil {
		m.Device.Release()
	}
	if m.Adapter != nil {
		m.Adapter.Release()
	}
	if m.Instance != nil {
		m.Instance.Release()
	}
}
