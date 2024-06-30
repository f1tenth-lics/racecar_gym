from setuptools import setup, find_packages

setup(
    name='racecar_gym',
    version='0.0.1',
    author='Joshua J. Damanik',
    author_email='joshuajdmk@gmail.com',
    description='A custom OpenAI Gym environment for racecar simulation.',
    packages=find_packages(include=['racecar_gym', 'racecar_gym.*']),
    install_requires=[
        'gym<0.21.0',
        'numpy',
        'rospy',
        'rospkg',
        'scipy'
    ],
    python_requires='>=3.6'
)
