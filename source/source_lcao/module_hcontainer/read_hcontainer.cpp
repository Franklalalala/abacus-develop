#include "read_hcontainer.h"

#include "source_io/sparse_matrix.h"
#include "source_io/csr_reader.h"
#include "hcontainer_funcs.h"

#include <fstream>

namespace hamilt
{

#ifdef __MPI
namespace
{
// Keep the <IJR>s whose rows (atom i) and columns (atom j) this rank owns.
// Layout as HContainer::get_ijr_info(): n_pairs, then per pair i, j, n_R, n_R x (rx, ry, rz).
std::vector<int> local_ijr_info(const std::vector<int>& ijrs, const Parallel_Orbitals* pv)
{
    std::vector<int> local(1, 0);
    if (ijrs.empty())
    {
        return local;
    }
    const int* p = ijrs.data();
    const int number_ap = *p++;
    for (int i = 0; i < number_ap; ++i)
    {
        const int atom_i = *p++;
        const int atom_j = *p++;
        const int number_R = *p++;
        if (pv->get_row_size(atom_i) > 0 && pv->get_col_size(atom_j) > 0)
        {
            local.push_back(atom_i);
            local.push_back(atom_j);
            local.push_back(number_R);
            local.insert(local.end(), p, p + 3 * number_R);
            ++local[0];
        }
        p += 3 * number_R;
    }
    return local;
}
} // namespace
#endif

/**
 * @brief Constructor of Read_HContainer
 * @attention ifs should be open outside of this interface
 */
template <typename T>
Read_HContainer<T>::Read_HContainer(hamilt::HContainer<T>* hcontainer,
                                        const std::string& filename,
                                        const int nlocal,
                                        const UnitCell* ucell)
    : _hcontainer(hcontainer), _filename(filename), _nlocal(nlocal), _ucell(ucell)
{
}

template <typename T>
void Read_HContainer<T>::read()
{
    // build atom index of col and row
    std::vector<int> atom_index_row;
    std::vector<int> atom_index_col;
    int natom = this->_ucell->nat;
    Parallel_Orbitals pv_serial;
    pv_serial.set_serial(this->_nlocal, this->_nlocal);
    pv_serial.set_atomic_trace(this->_ucell->get_iat2iwt(), this->_ucell->nat, this->_nlocal);
    for (int iat = 0; iat < natom; ++iat)
    {
        int row_size = pv_serial.get_row_size(iat);
        int col_size = pv_serial.get_col_size(iat);
        for (int i = 0; i < row_size; ++i)
        {
            atom_index_row.push_back(iat);
        }
        for (int j = 0; j < col_size; ++j)
        {
            atom_index_col.push_back(iat);
        }
    }
    //
    hamilt::HContainer<T> hcontainer_serial(&pv_serial);

#ifdef __MPI
    if(GlobalV::MY_RANK == 0)
    {
#endif
    ModuleIO::csrFileReader<T> csr(this->_filename);
    int step = csr.getStep();
    int matrix_dimension = csr.getMatrixDimension();
    int r_number = csr.getNumberOfR();

    //construct serial hcontainer firstly
    // prepare atom index mapping from csr row/col to atom index
    for (int i = 0; i < r_number; i++)
    {
        std::vector<int> RCoord = csr.getRCoordinate(i);
        ModuleIO::SparseMatrix<T> sparse_matrix = csr.getMatrix(i);
        for (const auto& element: sparse_matrix.getElements())
        {
            int row = element.first.first;
            int col = element.first.second;
            T value = element.second;
           

            //insert into hcontainer
            int atom_i = atom_index_row[row];
            int atom_j = atom_index_col[col];
            auto* ij_pair = hcontainer_serial.find_pair(atom_i, atom_j);
            if(ij_pair == nullptr)
            {
                //insert new pair
                hamilt::AtomPair<T> new_pair(atom_i, atom_j, RCoord[0], RCoord[1], RCoord[2], &pv_serial);
                hcontainer_serial.insert_pair(new_pair);
            }
            else
            {
                if(ij_pair->find_R(RCoord[0], RCoord[1], RCoord[2]) == -1)
                {
                    //insert new R
                    hamilt::AtomPair<T> new_pair(atom_i, atom_j, RCoord[0], RCoord[1], RCoord[2], &pv_serial);
                    hcontainer_serial.insert_pair(new_pair);
                }
            }
        }
    }
    hcontainer_serial.allocate(nullptr, true);
    // second loop, add values into hcontainer
    for (int i = 0; i < r_number; i++)
    {
        std::vector<int> RCoord = csr.getRCoordinate(i);
        ModuleIO::SparseMatrix<T> sparse_matrix = csr.getMatrix(i);
        for (const auto& element: sparse_matrix.getElements())
        {
            int row = element.first.first;
            int col = element.first.second;
            T value = element.second;

            //insert into hcontainer
            int atom_i = atom_index_row[row];
            int atom_j = atom_index_col[col];
            auto* matrix = hcontainer_serial.find_matrix(atom_i, atom_j, RCoord[0], RCoord[1], RCoord[2]);
            matrix->add_element(row - pv_serial.atom_begin_row[atom_i],
                                col - pv_serial.atom_begin_col[atom_j],
                                value);
        }
    }
#ifdef __MPI
}
    // thirdly, distribute hcontainer_serial to the parallel hcontainer.
    // A target that already holds atom pairs (the H(R) or DM(R) of this run) keeps its layout
    // and only receives values. Inserting every <IJR> of the file instead made each rank
    // construct AtomPairs it does not own in the 2D block-cyclic distribution (AtomPair throws
    // "Atom-pair not belong this process") and added pairs or R outside the run's neighbour
    // list. Blocks of the file that the target does not hold are skipped in
    // HTransPara::unpack_data. An empty target is built from the file, each rank keeping only
    // the pairs it owns.
    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int local_pairs = this->_hcontainer->size_atom_pairs();
    int max_pairs = 0;
    MPI_Allreduce(&local_pairs, &max_pairs, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (max_pairs == 0)
    {
        std::vector<int> para_ijrs;
        if (my_rank == 0)
        {
            para_ijrs = hcontainer_serial.get_ijr_info();
        }
        int ijrs_size = para_ijrs.size();
        MPI_Bcast(&ijrs_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        para_ijrs.resize(ijrs_size);
        MPI_Bcast(para_ijrs.data(), ijrs_size, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<int> local_ijrs = local_ijr_info(para_ijrs, this->_hcontainer->get_paraV());
        this->_hcontainer->insert_ijrs(&local_ijrs);
        this->_hcontainer->allocate();
    }
    else
    {
        // the transfer adds into the target; blocks absent from the file stay zero
        this->_hcontainer->set_zero();
    }
    // gather values from serial_rank to Parallels
    transferSerial2Parallels(hcontainer_serial, this->_hcontainer, 0);
#else
    std::vector<int> para_ijrs = hcontainer_serial.get_ijr_info();
    this->_hcontainer->insert_ijrs(&para_ijrs);
    this->_hcontainer->allocate();
    this->_hcontainer->add(hcontainer_serial);
#endif

}

template class Read_HContainer<double>;
template class Read_HContainer<std::complex<double>>;

} // namespace hamilt
