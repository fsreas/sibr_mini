#include "myData.hpp"

using namespace std;

namespace myViewer {

    bool    myMesh::load(const std::string& filename, const std::string& dataset_path )
	{
		// Does the file exists?
		if (!sibr::fileExists(filename)) {
			std::cout << "Error: can't load mesh '" << filename << "." << std::endl;
			return false;
		}
		Assimp::Importer	importer;
		//importer.SetPropertyBool(AI_CONFIG_PP_FD_REMOVE, true); // cause Assimp to remove all degenerated faces as soon as they are detected
		const aiScene* scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_FindDegenerates);

		if (!scene)
		{
			std::cout << "error: can't load mesh '" << filename
				<< "' (" << importer.GetErrorString() << ")." << std::endl;
			return false;
		}

		// check for texture
		aiMaterial *material;
		if( scene->mNumMaterials > 0 ) {
			material = scene->mMaterials[0];
			aiString Path;
			if(material->GetTexture(aiTextureType_DIFFUSE, 0, &Path, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS ) {
				_textureImageFileName = Path.data;
				std::cerr << "Texture name " << _textureImageFileName << std::endl;
			}

		}

		if (scene->mNumMeshes == 0)
		{
			std::cout << "error: the loaded model file ('" << filename
				<< "') contains zero or more than one mesh. Number of meshes : " << scene->mNumMeshes << std::endl;
			return false;
		}

		auto convertVec = [](const aiVector3D& v) { return Vector3f(v.x, v.y, v.z); };
		_triangles.clear();

		uint offsetVertices = 0;
		uint offsetFaces = 0;
		uint matId = 0;
		std::map<std::string, int> matName2Id;
		Eigen::Matrix3f converter;
		converter <<
			1, 0, 0,
			0, 1, 0,
			0, 0, 1;

		for (uint meshId = 0; meshId < scene->mNumMeshes; ++meshId) {
			const aiMesh* mesh = scene->mMeshes[meshId];

			_vertices.resize(offsetVertices + mesh->mNumVertices);
			for (uint i = 0; i < mesh->mNumVertices; ++i)
				_vertices[offsetVertices + i] = converter * convertVec(mesh->mVertices[i]);


			if (mesh->HasVertexColors(0) && mesh->mColors[0])
			{
				_colors.resize(offsetVertices + mesh->mNumVertices);
				for (uint i = 0; i < mesh->mNumVertices; ++i)
				{
					_colors[offsetVertices + i] = Vector3f(
						mesh->mColors[0][i].r,
						mesh->mColors[0][i].g,
						mesh->mColors[0][i].b);
				}
			}

			if (mesh->HasNormals())
			{
				_normals.resize(offsetVertices + mesh->mNumVertices);
				for (uint i = 0; i < mesh->mNumVertices; ++i) {
					_normals[offsetVertices + i] = converter * convertVec(mesh->mNormals[i]);
				}

			}

			if (mesh->HasTextureCoords(0))
			{
				_texcoords.resize(offsetVertices + mesh->mNumVertices);
				for (uint i = 0; i < mesh->mNumVertices; ++i)
					_texcoords[offsetVertices + i] = convertVec(mesh->mTextureCoords[0][i]).xy();
				// TODO: make a clean function
				std::string texFileName = dataset_path + "/capreal/" + _textureImageFileName;
				if( !fileExists(texFileName))
					texFileName = parentDirectory(parentDirectory(dataset_path)) + "/capreal/" + _textureImageFileName;
				if( !fileExists(texFileName))
					texFileName = parentDirectory(dataset_path) + "/capreal/" + _textureImageFileName;

				if (!mesh->HasVertexColors(0) && fileExists(texFileName)) {
					// Sample the texture
					sibr::ImageRGB texImg;
					texImg.load(texFileName);
					std::cout << "Computing vertex colors ..";
					_colors.resize(offsetVertices + mesh->mNumVertices);
					for (uint ci = 0; ci < mesh->mNumVertices; ++ci)
					{
						Vector2f uv = _texcoords[offsetVertices + ci];
						Vector3ub col = texImg((uv[0]*texImg.w()), uint((1-uv[1])*texImg.h()));
						_colors[offsetVertices + ci] = Vector3f(float(col[0]) / 255.0, float(col[1]) / 255.0, float(col[2]) / 255.0);
					}
					std::cout << "Done." << std::endl;
				}
			}
			if (meshId == 0) {
				std::cout << "Mesh contains: colors: " << mesh->HasVertexColors(0)
					<< ", normals: " << mesh->HasNormals()
					<< ", texcoords: " << mesh->HasTextureCoords(0) << std::endl;
			}

			_triangles.reserve(offsetFaces + mesh->mNumFaces);
			for (uint i = 0; i < mesh->mNumFaces; ++i)
			{
				const aiFace* f = &mesh->mFaces[i];
				if (f->mNumIndices != 3)
					std::cout << "warning: discarding a face (not a triangle, num indices: "
					<< f->mNumIndices << ")" << std::endl;
				else
				{
					Vector3u tri = Vector3u(offsetVertices + f->mIndices[0], offsetVertices + f->mIndices[1], offsetVertices + f->mIndices[2]);
					if (tri[0] < 0 || tri[0] >= _vertices.size()
						|| tri[1] < 0 || tri[1] >= _vertices.size()
						|| tri[2] < 0 || tri[2] >= _vertices.size())
						std::cout << "face num [" << i << "] contains invalid vertex id(s)" << std::endl;
					else {
						_triangles.push_back(tri);
					}

				}
			}

			offsetFaces = (uint)_triangles.size();
			offsetVertices = (uint)_vertices.size();

		}

		_meshPath = filename;

		std::cout << "Mesh '" << filename << " successfully loaded. " << scene->mNumMeshes << " meshes were loaded with a total of "
			<< " (" << _triangles.size() << ") faces and "
			<< " (" << _vertices.size() << ") vertices detected. Init GL ..." << std::endl;
		std::cout << "Init GL mesh complete " << std::endl;

		_gl.dirtyBufferGL = true;
		return true;
	};
	
    // init the GaussianScene
    GaussianScene::GaussianScene(string& model_path) {  
        // load cfg_args
        string cfgLine;
        ifstream cfgFile(model_path + "/cfg_args");
        getline(cfgFile, cfgLine);

        // get sh_degree
        auto rng = findArg(cfgLine, "sh_degree");
        _sh_degree = stoi(cfgLine.substr(rng.first, rng.second - rng.first));
        // get plyfile path
        string plyfile = model_path;
        if (plyfile.back() != '/')
            plyfile += "/";
        plyfile += "point_cloud/" + findLargestNumberedSubdirectory(plyfile) + "/point_cloud.ply";
        // get bg_color 
        bool white_bg = cfgLine.substr(rng.first, rng.second - rng.first).find("True") != -1;
        // get camera
        string jsonPath = model_path + "/cameras.json";
        _cameras = loadCameras(jsonPath);
        _mesh.load(plyfile, model_path);

        // load plyfile
        vector<Vector3f> pos;
        vector<Vector4f> rot;
        vector<Vector3f> scale;
        vector<float> opacity;
        vector<SHs<3>> shs;
        if (_sh_degree == 1) {
            count = loadPly<1>(plyfile, pos, shs, opacity, scale, rot);
        }else if (_sh_degree == 2) {
            count = loadPly<2>(plyfile, pos, shs, opacity, scale, rot);
        }else if (_sh_degree == 3) {
            count = loadPly<3>(plyfile, pos, shs, opacity, scale, rot);
        }

        // Allocate and fill the GPU data
        CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&pos_cuda, sizeof(Vector3f) * P));
        CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos.data(), sizeof(Vector3f) * P, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rot_cuda, sizeof(Vector4f) * P));
        CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda, rot.data(), sizeof(Vector4f) * P, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&shs_cuda, sizeof(SHs<3>) * P));
        CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda, shs.data(), sizeof(SHs<3>) * P, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&opacity_cuda, sizeof(float) * P));
        CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, opacity.data(), sizeof(float) * P, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&scale_cuda, sizeof(Vector3f) * P));
        CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda, scale.data(), sizeof(Vector3f) * P, cudaMemcpyHostToDevice));

        // Create space for view parameters
        CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&view_cuda, sizeof(sibr::Matrix4f)));
        CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&proj_cuda, sizeof(sibr::Matrix4f)));
        CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cam_pos_cuda, 3 * sizeof(float)));
        CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
        CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * P * sizeof(int)));

        // Set background color
        float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
        CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

        // GL init
		glCreateBuffers(1, &meanBuffer);
		glCreateBuffers(1, &rotBuffer);
		glCreateBuffers(1, &scaleBuffer);
		glCreateBuffers(1, &alphaBuffer);
		glCreateBuffers(1, &colorBuffer);
		glNamedBufferStorage(meanBuffer, count * 3 * sizeof(float), (float*)pos.data(), 0);
		glNamedBufferStorage(rotBuffer, count * 4 * sizeof(float), (float*)rot.data(), 0);
		glNamedBufferStorage(scaleBuffer, count * 3 * sizeof(float), (float*)scale.data(), 0);
		glNamedBufferStorage(alphaBuffer, count * sizeof(float), opacity.data(), 0);
		glNamedBufferStorage(colorBuffer, count * sizeof(float) * 48, (float*)shs.data(), 0);

        // surface renderer(needed???)
        // ...

        // Create GL buffer ready for CUDA/GL interop
        glCreateBuffers(1, &imageBuffer);
        glNamedBufferStorage(imageBuffer, _render_width * _render_height* 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

        bool useInterop = true;
        if (cudaPeekAtLastError() != cudaSuccess){
            std::cout << "[Viewer] --  INFOS  --:\t" << "A CUDA error occurred in setup:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
        }
        cudaGraphicsGLRegisterBuffer(&imageBufferCuda, imageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
        useInterop &= (cudaGetLastError() == cudaSuccess);

        geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
        binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
        imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);
    }

    // load the camera from the given json file
    vector<Camera> GaussianScene::loadCameras(const string& jsonPath) {
        vector<Camera> cameras;

        ifstream jsonFile(jsonPath);
        picojson::value myJson;
        picojson::set_last_error(std::string());
        std::string err = picojson::parse(myJson, jsonFile);
        if (!err.empty()) {
            picojson::set_last_error(err);
            jsonFile.setstate(std::ios::failbit);
        }

        picojson::array& frames = myJson.get<picojson::array>();

        for (size_t i = 0; i < frames.size(); ++i)
        {
            int id = frames[i].get("id").get<double>();
            std::string imgname = frames[i].get("img_name").get<std::string>();
            int width = frames[i].get("width").get<double>();
            int height = frames[i].get("height").get<double>();
            float fy = frames[i].get("fy").get<double>();
            float fx = frames[i].get("fx").get<double>();

            // new a camera object
            Camera myCamera;
            myCamera.fy = fy;
            myCamera.fx = fx;
            myCamera.cx = 0.0f;
            myCamera.cy = 0.0f;
            myCamera.width = width;
            myCamera.height = height;
            myCamera.id = id;
            myCamera.name = imgname;


            picojson::array& pos = frames[i].get("position").get<picojson::array>();
            myViewer::Vector3f position(pos[0].get<double>(), pos[1].get<double>(), pos[2].get<double>());
            picojson::array& rot = frames[i].get("rotation").get<picojson::array>();
            Eigen::Matrix3f orientation;
            for (int i = 0; i < 3; i++)
            {
                picojson::array& row = rot[i].get<picojson::array>();
                for (int j = 0; j < 3; j++)
                {
                    orientation(i, j) = row[j].get<double>();
                }
            }
            orientation.col(1) = -orientation.col(1);
            orientation.col(2) = -orientation.col(2);

            myCamera.position = position;
            myCamera.rotation = Eigen::Quaternionf(orientation);
            myCamera.znear = 0.01f;
            myCamera.zfar = 1000.0f;

            cameras.push_back(myCamera);
        }
        return cameras;
    }

    // Load the Gaussians from the given file.
    template<int D> int 
    GaussianScene::loadPly(const string& filename,
        std::vector<Vector3f>& pos,
        std::vector<SHs<3>>& shs,
        std::vector<float>& opacities,
        std::vector<Vector3f>& scales,
        std::vector<Vector4f>& rot)
    {
        std::ifstream infile(filename, std::ios_base::binary);

        if (!infile.good())
            std::cout << "[Viewer] --  INFOS  --:\t" << "Unable to find model's PLY file, attempted:\n" << filename << std::endl;

        // "Parse" header (it has to be a specific format anyway)
        std::string buff;
        std::getline(infile, buff);
        std::getline(infile, buff);

        std::string dummy;
        std::getline(infile, buff);
        std::stringstream ss(buff);
        int count;
        ss >> dummy >> dummy >> count;

        // Output number of Gaussians contained
        std::cout << "[Viewer] --  INFOS  --:\t" << "Loading " << count << " Gaussian splats" << std::endl;

        while (std::getline(infile, buff))
            if (buff.compare("end_header") == 0)
                break;

        // Read all Gaussians at once (AoS)
        std::vector<GaussianPoint<D>> points(count);
        infile.read((char*)points.data(), count * sizeof(GaussianPoint<D>));

        // Resize our SoA data
        pos.resize(count);
        shs.resize(count);
        scales.resize(count);
        rot.resize(count);
        opacities.resize(count);

        // Gaussians are done training, they won't move anymore. Arrange
        // them according to 3D Morton order. This means better cache
        // behavior for reading Gaussians that end up in the same tile 
        // (close in 3D --> close in 2D).
        myViewer::Vector3f minn(FLT_MAX, FLT_MAX, FLT_MAX);
        myViewer::Vector3f maxx = -minn;
        for (int i = 0; i < count; i++)
        {
            maxx = maxx.cwiseMax(points[i].pos);
            minn = minn.cwiseMin(points[i].pos);
        }
        std::vector<std::pair<uint64_t, int>> mapp(count);
        for (int i = 0; i < count; i++)
        {
            myViewer::Vector3f rel = (points[i].pos - minn).array() / (maxx - minn).array();
            myViewer::Vector3f scaled = ((float((1 << 21) - 1)) * rel);
            myViewer::Vector3i xyz = scaled.cast<int>();

            uint64_t code = 0;
            for (int i = 0; i < 21; i++) {
                code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
                code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
                code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
            }

            mapp[i].first = code;
            mapp[i].second = i;
        }
        auto sorter = [](const std::pair < uint64_t, int>& a, const std::pair < uint64_t, int>& b) {
            return a.first < b.first;
        };
        std::sort(mapp.begin(), mapp.end(), sorter);

        // Move data from AoS to SoA
        int SH_N = (D + 1) * (D + 1);
        for (int k = 0; k < count; k++)
        {
            int i = mapp[k].second;
            pos[k] = points[i].pos;

            // Normalize quaternion
            float length2 = 0;
            for (int j = 0; j < 4; j++)
                length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
            float length = sqrt(length2);
            for (int j = 0; j < 4; j++)
                rot[k].rot[j] = points[i].rot.rot[j] / length;

            // Exponentiate scale
            for(int j = 0; j < 3; j++)
                scales[k].scale[j] = exp(points[i].scale.scale[j]);

            // Activate alpha
            opacities[k] = sigmoid(points[i].opacity);

            shs[k].shs[0] = points[i].shs.shs[0];
            shs[k].shs[1] = points[i].shs.shs[1];
            shs[k].shs[2] = points[i].shs.shs[2];
            for (int j = 1; j < SH_N; j++)
            {
                shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
                shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
                shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
            }
        }
        return count;
    }

    // Destructor
    GaussianScene::~GaussianScene() {
        cudaFree(pos_cuda);
        cudaFree(rot_cuda);
        cudaFree(scale_cuda);
        cudaFree(opacity_cuda);
        cudaFree(shs_cuda);
    
        cudaFree(view_cuda);
        cudaFree(proj_cuda);
        cudaFree(cam_pos_cuda);
        cudaFree(background_cuda);
        cudaFree(rect_cuda);
    
        cudaGraphicsUnregisterResource(imageBufferCuda);

        glDeleteBuffers(1, &imageBuffer);
    
        if (geomPtr)
            cudaFree(geomPtr);
        if (binningPtr)
            cudaFree(binningPtr);
        if (imgPtr)
            cudaFree(imgPtr);
    
        delete _copyRenderer;
    }

}