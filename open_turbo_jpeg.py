from turbojpeg import TJPF_RGB, TurboJPEG

jpeg = TurboJPEG()

def open_turbo_jpeg(filepath):
    with open(filepath, "rb") as f:
        image = jpeg.decode(f.read(), TJPF_RGB)
    return image