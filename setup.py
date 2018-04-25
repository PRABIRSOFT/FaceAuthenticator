from setuptools import setup

setup(name='face_authenticator',
      version='0.2',
      description='Human face detection and authentication tool',
      url='http://github.com/prabirsoft/FaceAuthenticator',
      author='Prabir Ghosh',
      author_email='mymail.prabir@gmail.com',
      license='MIT',
      packages=['face_authenticator'],
      install_requires=["opencv-python","opencv-contrib-python"],
      include_package_data=True,
      zip_safe=False)
