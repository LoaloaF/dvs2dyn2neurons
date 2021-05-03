import Dynapse1Utils as ut

device_name = "dynapse1"
store, gui_process = ut.open_dynapse1(device_name)
model = getattr(store, device_name)

while True:
    pass