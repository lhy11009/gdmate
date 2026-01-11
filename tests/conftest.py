import pyvista as pv

def pytest_sessionstart(session):

    # This launches a virtual framebuffer so VTK thinks a display exists.
    # and prevents segmentation fault when running in Cl
    pv.start_xvfb()
