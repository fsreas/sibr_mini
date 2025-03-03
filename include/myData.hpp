/* this is a 3d gaussian data loader*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <regex>
#include <picojson/picojson.hpp>
#include <thread>
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

using namespace std;

# define CUDA_SAFE_CALL_ALWAYS(A) \
A; \
cudaDeviceSynchronize(); \
if (cudaPeekAtLastError() != cudaSuccess) \
std::cout << "[Viewer] --  INFOS  --:\t" << cudaGetErrorString(cudaGetLastError());

#if DEBUG || _DEBUG
# define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
# define CUDA_SAFE_CALL(A) A
#endif


namespace myViewer {
    
    typedef	Eigen::Matrix<float, 1,			1,Eigen::DontAlign>			Vector1f;
    typedef	Eigen::Matrix<int, 1,			1,Eigen::DontAlign>			Vector1i;

    typedef	Eigen::Matrix<unsigned, 2,		1,Eigen::DontAlign>			Vector2u;
    typedef	Eigen::Matrix<unsigned char, 2,	1,Eigen::DontAlign>			Vector2ub;
    typedef	Eigen::Matrix<int, 2,			1,Eigen::DontAlign>			Vector2i;
    typedef	Eigen::Matrix<float, 2,			1,Eigen::DontAlign>			Vector2f;
    typedef	Eigen::Matrix<double, 2,		1,Eigen::DontAlign>			Vector2d;

    typedef	Eigen::Matrix<unsigned, 3,		1,Eigen::DontAlign>			Vector3u;
    typedef	Eigen::Matrix<unsigned char, 3,	1,Eigen::DontAlign>			Vector3ub;
    typedef	Eigen::Matrix<unsigned short int, 3, 1,Eigen::DontAlign>	Vector3s;
    typedef	Eigen::Matrix<int, 3,			1,Eigen::DontAlign>			Vector3i;
    typedef	Eigen::Matrix<float, 3,			1,Eigen::DontAlign>			Vector3f;
    typedef	Eigen::Matrix<double, 3,		1,Eigen::DontAlign>			Vector3d;

    typedef	Eigen::Matrix<unsigned, 4,		1,Eigen::DontAlign>			Vector4u;
    typedef	Eigen::Matrix<unsigned char, 4,	1,Eigen::DontAlign>			Vector4ub;
    typedef	Eigen::Matrix<int, 4,			1,Eigen::DontAlign>			Vector4i;
    typedef	Eigen::Matrix<float, 4,			1,Eigen::DontAlign>			Vector4f;
    typedef	Eigen::Matrix<double, 4,		1,Eigen::DontAlign>			Vector4d;

    class myMesh
    {
    public:
        bool	load(const std::string& filename, const std::string& dataset_path);
    
        const std::vector<Vector3f>&	vertices() const { return _vertices; }
        const std::vector<Vector3u>&	triangles() const { return _triangles; }
        const std::vector<Vector3f>&	normals() const { return _normals; }
        const std::vector<Vector3f>&	colors() const { return _colors; }
        const std::vector<Vector3f>&	texcoords() const { return _texcoords; }
        const std::string&				textureImageFileName() const { return _textureImageFileName; }

    private:
        std::string	_textureImageFileName;
        std::vector<Vector3f> _vertices;
        std::vector<Vector3u> _triangles;
        std::vector<Vector3f> _normals;
        std::vector<Vector3f> _colors;
        std::vector<Vector3f> _texcoords;
    };

    template<int sh_degree>
    struct SHs
    {
        float shs[(sh_degree+1)*(sh_degree+1)*3];
    };

    template<int sh_degree>
    struct GaussianPoint
    {
        Vector3f pos;
        float n[3];
        SHs<sh_degree> shs;
        float opacity;
        Vector3f scale;
        Vector4f rot;
    };

    struct Camera
    {
        float fy;
        float fx;
        float cx;
        float cy;
        int width;
        int height;
        int id;

        string name;
        Vector3f position;
        Eigen::Quaternionf rotation;
        float znear;
        float zfar;

    };

    class GaussianScene {
        public:

        GaussianScene(string& model_path);
        vector<Camera> loadCameras(const string& jsonPath);
        template<int D> int loadPly(const string& filename,
            std::vector<Vector3f>& pos,
            std::vector<SHs<3>>& shs,
            std::vector<float>& opacities,
            std::vector<Vector3f>& scales,
            std::vector<Vector4f>& rot);
        myMesh getMesh() { return _mesh; }
        void render(Camera& view);

        ~GaussianScene();

        protected:
            int _render_width = 800;
            int _render_height = 600;
            int _sh_degree;   
            vector<Camera> _cameras;
            myMesh _mesh;

            // cuda part
            int count;
            float* pos_cuda;
            float* rot_cuda;
            float* scale_cuda;
            float* opacity_cuda;
            float* shs_cuda;
            int* rect_cuda;

            float* view_cuda;
            float* proj_cuda;
            float* cam_pos_cuda;
            float* background_cuda;

            // gl_buffer part
            int _num_gaussians;
            GLuint meanBuffer;
            GLuint rotBuffer;
            GLuint scaleBuffer;
            GLuint alphaBuffer;
            GLuint colorBuffer;

            // // texture part
            // GLuint idTexture;
            // GLuint colorTexture;
            // GLuint depthBuffer;
            // GLuint fbo;
            // int resX, resY;
    
            // GLShader			_shader; ///< Color shader.
            // GLParameter			_paramMVP; ///< MVP uniform.
            // GLParameter			_paramCamPos;
            // GLParameter			_paramLimit;
            // GLParameter			_paramStage;
            // GLuint clearProg;
            // GLuint clearShader;

            GLuint imageBuffer;
		    cudaGraphicsResource_t imageBufferCuda;

            size_t allocdGeom = 0, allocdBinning = 0, allocdImg = 0;
            void* geomPtr = nullptr, * binningPtr = nullptr, * imgPtr = nullptr;
            std::function<char* (size_t N)> geomBufferFunc, binningBufferFunc, imgBufferFunc;
    
            
    };


}