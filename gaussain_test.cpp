#include <iostream>
#include <string>
#include <fstream>
#include <regex>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <myShader.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "myData.hpp"
// #include "myRaycaster.hpp"

using namespace std;
// --------------------------- utilized functions ----------------------------

// callback function for resize the window
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

// add process
void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// --------------------------- main function -----------------------------------
int main() {

    // ------------- init glfw ---------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // ------------- init glad ---------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // ------------- create window ---------------
    GLFWwindow* window = glfwCreateWindow(800, 600, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glViewport(0, 0, 800, 600);

    // ------------- load data ---------------
    string model_path = "model/garden";
    auto myScene = std::make_shared<myViewer::GaussianScene>(model_path);

    // -------------- set the raycaster ---------------
    // auto myRC = make_shared<Raycaster>();
    // myRC->init();
    // myRC->addMesh(myScene->getMesh());

    // ------- register the callback function ----------
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    // more callback functions can be added here

    // ------------- set the Shader ---------------
    Shader ourShader("shader/vertexShader.vs", "shader/fragmentShader.fs");

    // ----------- set the object here ---------------
    unsigned int VBO;
    unsigned int VAO;
    unsigned int EBO;

    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glGenVertexArrays(1, &VAO);

    // --------- bind these object to buffer -----------
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

    // --------- transport the data to buffer -----------
    // glBufferData(GL_ARRAY_BUFFER, sizeof(xxxxx), xxxxx, GL_STATIC_DRAW);
    // glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(xxxxx), xxxxx, GL_STATIC_DRAW);

    // --------- set the vertex attributes pointer ------
    // to-do

    // --------- set the uniform variable --------
    // to-do

    // ---------------- main loop ----------------
    // while(!glfwWindowShouldClose(window))
    // {
    //     processInput(window);

    //     // background color
    //     glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    //     glClear(GL_COLOR_BUFFER_BIT);

    //     // activate the program and draw
    //     ourShader.use();

    //     glBindVertexArray(VAO);
    //     glDrawArrays(
    //         GL_TRIANGLES,   //type
    //         0,  // begin index
    //         xxxxx// to-do
    //     );

    //     glfwSwapBuffers(window);
    //     glfwPollEvents();
    // }

    // ------------- release the resources ---------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glfwTerminate();
};