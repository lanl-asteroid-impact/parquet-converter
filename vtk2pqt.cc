/*
 * Copyright (c) 2025 Triad National Security, LLC, as operator of Los Alamos
 * National Laboratory with the U.S. Department of Energy/National Nuclear
 * Security Administration. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of TRIAD, Los Alamos National Laboratory, LANL, the
 *    U.S. Government, nor the names of its contributors may be used to endorse
 *    or promote products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <arrow/io/file.h>
#include <parquet/stream_writer.h>

#include <vtkCellArrayIterator.h>
#include <vtkCellDataToPointData.h>
#include <vtkFloatArray.h>
#include <vtkIdList.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>

#include <cstdint>
#include <filesystem>
#include <string>

namespace {

class Iterator {
 public:
  explicit Iterator(vtkUnstructuredGrid* grid);
  ~Iterator() = default;

  void SeekToFirst() { i_ = 0; }
  bool Valid() const { return i_ >= 0 && i_ < n_; }
  void Next() { i_++; }

  float rho() const { return rho_[i_]; }
  float prs() const { return prs_[i_]; }
  float tev() const { return tev_[i_]; }
  float xdt() const { return xdt_[i_]; }
  float ydt() const { return ydt_[i_]; }
  float zdt() const { return zdt_[i_]; }
  float snd() const { return snd_[i_]; }
  float grd() const { return grd_[i_]; }
  float mat() const { return mat_[i_]; }
  float v02() const { return v02_[i_]; }
  float v03() const { return v03_[i_]; }

  float x() const { return point_loc_[i_ * 3]; }
  float y() const { return point_loc_[i_ * 3 + 1]; }
  float z() const { return point_loc_[i_ * 3 + 2]; }
  int current_point() const { return i_; }

 private:
  float* point_loc_;
  int n_;  // Total number of elements
  float* rho_;
  float* prs_;
  float* tev_;
  float* xdt_;
  float* ydt_;
  float* zdt_;
  float* snd_;
  float* grd_;
  float* mat_;
  float* v02_;
  float* v03_;
  int i_;
};

Iterator::Iterator(vtkUnstructuredGrid* grid) {
  point_loc_ =
      vtkFloatArray::FastDownCast(grid->GetPoints()->GetData())->GetPointer(0);
  vtkPointData* const pointData = grid->GetPointData();
  n_ = grid->GetNumberOfPoints();
#define GET_POINTER(d, f) \
  vtkFloatArray::FastDownCast(d->GetAbstractArray(f))->GetPointer(0)
  rho_ = GET_POINTER(pointData, "rho");
  prs_ = GET_POINTER(pointData, "prs");
  tev_ = GET_POINTER(pointData, "tev");
  xdt_ = GET_POINTER(pointData, "xdt");
  ydt_ = GET_POINTER(pointData, "ydt");
  zdt_ = GET_POINTER(pointData, "zdt");
  snd_ = GET_POINTER(pointData, "snd");
  grd_ = GET_POINTER(pointData, "grd");
  mat_ = GET_POINTER(pointData, "mat");
  v02_ = GET_POINTER(pointData, "v02");
  v03_ = GET_POINTER(pointData, "v03");
#undef GET_POINTER
  i_ = 0;
}

struct ParquetWriterOptions {
  ParquetWriterOptions() = default;
};

class ParquetWriter {
 public:
  ParquetWriter(const ParquetWriterOptions& options,
                std::shared_ptr<arrow::io::OutputStream> file);
  void Append(int64_t pointId, Iterator* it);
  void Finish();
  ~ParquetWriter();

  // No copying allowed
  ParquetWriter(const ParquetWriter&) = delete;
  ParquetWriter& operator=(const ParquetWriter& other) = delete;

 private:
  parquet::StreamWriter* writer_;
};

std::shared_ptr<parquet::schema::GroupNode> GetSchema() {
  parquet::schema::NodeVector fields;
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "point_id", parquet::Repetition::REQUIRED, parquet::Type::INT64,
      parquet::ConvertedType::INT_64));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "x", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "y", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "z", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "rho", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "prs", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "tev", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "xdt", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "ydt", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "zdt", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "snd", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "grd", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "mat", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "v02", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "v03", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  return std::static_pointer_cast<parquet::schema::GroupNode>(
      parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED,
                                       fields));
}

ParquetWriter::ParquetWriter(const ParquetWriterOptions& options,
                             std::shared_ptr<arrow::io::OutputStream> file) {
  parquet::WriterProperties::Builder builder;
  builder.compression(parquet::Compression::SNAPPY);
  builder.encoding("point_id", parquet::Encoding::DELTA_BINARY_PACKED);
  builder.encoding(parquet::Encoding::PLAIN);
  builder.disable_dictionary("point_id");
  writer_ = new parquet::StreamWriter(parquet::ParquetFileWriter::Open(
      std::move(file), GetSchema(), builder.build()));
}

void ParquetWriter::Append(int64_t pointId, Iterator* it) {
  *writer_ << pointId << it->x() << it->y() << it->z() << it->rho() << it->prs()
           << it->tev() << it->xdt() << it->ydt() << it->zdt() << it->snd()
           << it->grd() << it->mat() << it->v02() << it->v03()
           << parquet::EndRow;
}

void ParquetWriter::Finish() {
  delete writer_;
  writer_ = nullptr;
}

ParquetWriter::~ParquetWriter() { delete writer_; }

class CellWriter {
 public:
  CellWriter(const ParquetWriterOptions& options,
             std::shared_ptr<arrow::io::OutputStream> file);
  void Append(int64_t cellId, int64_t pointId);
  void Finish();
  ~CellWriter();

  // No copying allowed
  CellWriter(const CellWriter&) = delete;
  void operator=(const CellWriter& other) = delete;

 private:
  parquet::StreamWriter* writer_;
};

std::shared_ptr<parquet::schema::GroupNode> CellSchema() {
  parquet::schema::NodeVector fields;
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "cell_id", parquet::Repetition::REQUIRED, parquet::Type::INT64,
      parquet::ConvertedType::INT_64));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "point_id", parquet::Repetition::REQUIRED, parquet::Type::INT64,
      parquet::ConvertedType::INT_64));
  return std::static_pointer_cast<parquet::schema::GroupNode>(
      parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED,
                                       fields));
}

CellWriter::CellWriter(const ParquetWriterOptions& options,
                       std::shared_ptr<arrow::io::OutputStream> file) {
  parquet::WriterProperties::Builder builder;
  builder.compression(parquet::Compression::SNAPPY);
  builder.encoding(parquet::Encoding::DELTA_BINARY_PACKED);
  builder.disable_dictionary();
  writer_ = new parquet::StreamWriter(parquet::ParquetFileWriter::Open(
      std::move(file), CellSchema(), builder.build()));
}

void CellWriter::Append(int64_t cellId, int64_t pointId) {
  *writer_ << cellId << pointId << parquet::EndRow;
}

void CellWriter::Finish() {
  delete writer_;
  writer_ = nullptr;
}

CellWriter::~CellWriter() { delete writer_; }

}  // namespace

struct Stats {
  vtkIdType nCells = 0;
  vtkIdType nPoints = 0;
};

void Process(const std::string& fname, CellWriter* cellWriter,
             ParquetWriter* writer, Stats* stats) {
  vtkNew<vtkXMLUnstructuredGridReader> reader;
  reader->SetFileName(fname.c_str());
  vtkNew<vtkCellDataToPointData> filter;
  filter->SetInputConnection(reader->GetOutputPort());
  filter->Update();
  vtkUnstructuredGrid* const grid =
      vtkUnstructuredGrid::SafeDownCast(filter->GetOutput());
  const vtkIdType cellOffset = stats->nCells;
  const vtkIdType pointOffset = stats->nPoints;
  {
    auto it = vtk::TakeSmartPointer(grid->GetCells()->NewIterator());
    it->GoToFirstCell();
    while (!it->IsDoneWithTraversal()) {
      vtkIdType cellId = it->GetCurrentCellId();
      vtkIdList* const points = it->GetCurrentCell();
      vtkIdType n = points->GetNumberOfIds();
      for (int i = 0; i < n; i++) {
        cellWriter->Append(cellOffset + cellId, pointOffset + points->GetId(i));
      }
      it->GoToNextCell();
    }
  }
  {
    Iterator it(grid);
    it.SeekToFirst();
    while (it.Valid()) {
      int pointId = it.current_point();
      writer->Append(pointOffset + pointId, &it);
      it.Next();
    }
  }
  stats->nPoints += grid->GetNumberOfPoints();
  stats->nCells += grid->GetNumberOfCells();
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " pv_insitu_N" << std::endl;
    return 1;
  }
  std::string dir = argv[1];
  std::cout << "Converting " << dir << "..." << std::endl;
  std::filesystem::path filename =
      std::filesystem::path(dir.c_str()).filename();
  std::shared_ptr<arrow::io::FileOutputStream> cellFile;
  PARQUET_ASSIGN_OR_THROW(
      cellFile, arrow::io::FileOutputStream::Open(dir + "_cells.parquet"))
  CellWriter cellWriter(ParquetWriterOptions(), cellFile);
  std::shared_ptr<arrow::io::FileOutputStream> file;
  PARQUET_ASSIGN_OR_THROW(file,
                          arrow::io::FileOutputStream::Open(dir + ".parquet"))
  ParquetWriter writer(ParquetWriterOptions(), file);
  Stats stats;
  char tmp[100];
  for (int i = 0; i < 512; i++) {
    snprintf(tmp, sizeof(tmp), "/%s_0_%d.vtu", filename.c_str(), i);
    std::string path = dir + tmp;
    std::cout << "Processing " << tmp + 1 << " ..." << std::endl;
    Process(path, &cellWriter, &writer, &stats);
  }
  std::cout << "Done!" << std::endl;
  cellWriter.Finish();
  writer.Finish();
  return 0;
}
