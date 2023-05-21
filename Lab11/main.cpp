//#include <glad/glad.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdio.h>
#include <malloc.h>

const unsigned int window_width = 512;
const unsigned int window_height = 512;
void initGL();
GLuint* bufferID;
void initBuffers(GLuint*&);
void transformBuffers(GLuint*);
void outputBuffers(GLuint*);

int main() {
	initGL();
	bufferID = (GLuint*)calloc(2, sizeof(GLuint));
	initBuffers(bufferID);
	transformBuffers(bufferID);
	outputBuffers(bufferID);
	glDeleteBuffers(2, bufferID);
	free(bufferID);
	glfwTerminate();

	return 0;
}

void initGL() {
	GLFWwindow* window;
	glfwWindowHint(GLFW_VISIBLE, 0);
	window = glfwCreateWindow(window_width, window_height, "Templatewindow", NULL, NULL);
	glfwMakeContextCurrent(window);
	int glewErr = glewInit();
}

const int N = 256;
GLuint genInitProg();

void initBuffers(GLuint*& bufferID) {
	glGenBuffers(2, bufferID);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferID[0]);
	glBufferData(GL_SHADER_STORAGE_BUFFER, N * sizeof(float), 0,
		GL_DYNAMIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferID[1]);
	glBufferData(GL_SHADER_STORAGE_BUFFER, N * sizeof(float), 0,
		GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID[0]);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferID[1]);

	GLuint csInitID = genInitProg();
	glUseProgram(csInitID);
	glDispatchCompute(N / 128, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
	glDeleteProgram(csInitID);
}

GLuint genInitProg()
{
	GLuint progHandle = glCreateProgram();
	GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
	const char* cpSrc[] = {
	"#version 430\n",
	"layout (local_size_x = 128, local_size_y = 1, local_size_z = 1) in; \
	layout(std430, binding = 0) buffer BufferA{float A[];};\
	layout(std430, binding = 1) buffer BufferB{float B[];};\
	void main() {\
	uint index = gl_GlobalInvocationID.x;\
	A[index]=0.1*float(index);\
	B[index]=0.2*float(index);\
	}"
	};
}

GLuint genTransformProg();

void transformBuffers(GLuint* bufferID)
{
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID[0]);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferID[1]);

	GLuint csTransformID = genTransformProg();
	glUseProgram(csTransformID);
	glDispatchCompute(N / 128, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

	glDeleteProgram(csTransformID);
}

GLuint genTransformProg() 
{
	GLuint progHandle = glCreateProgram();
	GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
	const char* cpSrc[] = {
	"#version 430\n",
	"layout (local_size_x = 128, local_size_y = 1, local_size_z = 1) in; \
	layout(std430, binding = 0) buffer BufferA{float A[];};\
	layout(std430, binding = 1) buffer BufferB{float B[];};\
	void main() {\
	uint index = gl_GlobalInvocationID.x;\
	A[index]=A[index]+B[index];\
	}"
	};
	return cpSrc[];
}