#include <Python.h>
#include <iostream>
#include <string>

class TFModel
{
public:
    TFModel();
    ~TFModel();

public:
    void generateMeshSingleView(std::string sketch_name);       // for single view(front or side) sketch2mesh prediction
    void generateMeshDualView(std::string sketch_name);         // for dual view sketch2mesh prediction
    void generateMesh(std::string sketch_name);                 // for whole model
    void generateVertLabelProb(std::string mesh_name);  // for mesh2feature: mesh vertex feature probability prediction

private:
    std::string module_dir = "./";
    std::string module_name = "sanihead";
    std::string func_name_pred_single = "predict_single";
    std::string func_name_pred_dual = "predict_dual";
    std::string func_name_pred_whole = "predict_whole";
    std::string func_name_pred_flc = "predict_flc";
    PyObject *p_name = nullptr, *p_module = nullptr, *p_dict = nullptr,
             *p_func_pred_single = nullptr, *p_instance_pred_single = nullptr,
             *p_func_pred_dual = nullptr, *p_instance_pred_dual = nullptr,
             *p_func_pred_whole = nullptr, *p_instance_pred_whole = nullptr,
             *p_func_pred_flc = nullptr, *p_instance_pred_flc = nullptr;
};

TFModel::TFModel()
{
    Py_Initialize();
    if (!Py_IsInitialized())
    {
        PyEval_InitThreads();
        fprintf(stderr, "Cannot initialize python runtime!\n");
        exit(1);
    }

    std::string s = "import sys\nsys.path.append(\"" + module_dir + "\")";
    PyRun_SimpleString(s.c_str());

    p_name = PyUnicode_DecodeFSDefault(module_name.c_str());
    p_module = PyImport_Import(p_name);
    if (!p_module)
    {
        fprintf(stderr, "Cannot open python file!\n");
        exit(1);
    }
    p_dict = PyModule_GetDict(p_module);
    if (!p_dict)
    {
        fprintf(stderr, "Cannot find dictionary!\n");
        exit(1);
    }

    p_func_pred_single = PyDict_GetItemString(p_dict, func_name_pred_single.c_str());
    if (!p_func_pred_single || !PyCallable_Check(p_func_pred_single))
    {
        fprintf(stderr, "Cannot find function \"%s\"\n", func_name_pred_single.c_str());
        exit(1);
    }

    p_func_pred_dual = PyDict_GetItemString(p_dict, func_name_pred_dual.c_str());
    if (!p_func_pred_dual || !PyCallable_Check(p_func_pred_dual))
    {
        fprintf(stderr, "Cannot find function \"%s\"\n", func_name_pred_dual.c_str());
        exit(1);
    }

    p_func_pred_whole = PyDict_GetItemString(p_dict, func_name_pred_whole.c_str());
    if (!p_func_pred_whole || !PyCallable_Check(p_func_pred_whole))
    {
        fprintf(stderr, "Cannot find function \"%s\"\n", func_name_pred_whole.c_str());
        exit(1);
    }

    p_func_pred_flc = PyDict_GetItemString(p_dict, func_name_pred_flc.c_str());
    if (!p_func_pred_flc || !PyCallable_Check(p_func_pred_flc))
    {
        fprintf(stderr, "Cannot find function \"%s\"\n", func_name_pred_flc.c_str());
        exit(1);
    }
}

void TFModel::generateMeshSingleView(std::string sketch_name)
{
    p_instance_pred_single = PyObject_CallFunction(p_func_pred_single, "s", sketch_name.c_str());
    if (!p_instance_pred_single)
    {
        fprintf(stderr, "Cannot create instance of \"%s\"\n", func_name_pred_single.c_str());
    }
}

void TFModel::generateMeshDualView(std::string sketch_name)
{
    p_instance_pred_dual = PyObject_CallFunction(p_func_pred_dual, "s", sketch_name.c_str());
    if (!p_instance_pred_dual)
    {
        fprintf(stderr, "Cannot create instance of \"%s\"\n", func_name_pred_dual.c_str());
    }
}

void TFModel::generateMesh(std::string sketch_name)
{
    p_instance_pred_whole = PyObject_CallFunction(p_func_pred_whole, "s", sketch_name.c_str());
    if (!p_instance_pred_whole)
    {
        fprintf(stderr, "Cannot create instance of \"%s\"\n", func_name_pred_whole.c_str());
    }
}

void TFModel::generateVertLabelProb(std::string mesh_name)
{
    p_instance_pred_flc = PyObject_CallFunction(p_func_pred_flc, "s", mesh_name.c_str());
    if (!p_instance_pred_flc)
    {
        fprintf(stderr, "Cannot create instance of \"%s\"\n", func_name_pred_flc.c_str());
    }
}

TFModel::~TFModel()
{
    Py_DECREF(p_name);
    Py_DECREF(p_module);
    Py_DECREF(p_dict);
    Py_DECREF(p_func_pred_single);
    Py_DECREF(p_func_pred_dual);
    Py_DECREF(p_func_pred_whole);
    Py_DECREF(p_func_pred_flc);
    Py_DECREF(p_instance_pred_single);
    Py_DECREF(p_instance_pred_dual);
    Py_DECREF(p_instance_pred_whole);
    Py_DECREF(p_instance_pred_flc);

    Py_Finalize();
    std::cout << "SAniHead destroyed!" << std::endl;
}
