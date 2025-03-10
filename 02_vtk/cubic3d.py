import numpy as np
import vtk


# Класс расчётной сетки
class CalcMesh:
    # Конструктор сетки size x size x size точек с шагом h по пространству
    def __init__(self, size, step):
        # 3D-сетка из расчётных точек
        self.nodes = np.mgrid[0:size-1:complex(size), 0:size-1:complex(size), 0:size-1:complex(size)]
        self.nodes *= step

        # Модельная скалярная величина распределена как-то вот так
        self.smth = np.power(self.nodes[0], 2) + np.power(self.nodes[1], 2) + np.power(self.nodes[2], 2)

        # Профиль скорости (но в данном случае сплошные нули)
        self.velocity = np.zeros(shape=(3, size, size, size), dtype=np.double)
        # self.velocity[2] = np.power(self.nodes[0] - self.nodes[1], 2)
        # self.velocity[0] = 5*self.nodes[0]
        # print(self.velocity[0][0])
        for i in range(5, size):
            self.velocity[0][i] = 1*pow(self.nodes[0][i] - self.nodes[1][i], 1)

    # Метод отвечает за выполнение для всей сетки шага по времени величиной tau
    def move(self, tau):
        # По сути метод просто двигает все точки c их текущими скоростями
        self.nodes += self.velocity * tau

    # Метод отвечает за запись текущего состояния сетки в снапшот в формате VTK
    def snapshot(self, snap_number):
        # Сетка в терминах VTK
        structuredGrid = vtk.vtkStructuredGrid()
        # Точки сетки в терминах VTK
        points = vtk.vtkPoints()

        # Скалярное поле на точках сетки
        smth = vtk.vtkDoubleArray()
        smth.SetName("smth")

        # Векторное поле на точках сетки
        vel = vtk.vtkDoubleArray()
        vel.SetNumberOfComponents(3)
        vel.SetName("vel")

        # Обходим все точки нашей расчётной сетки
        # Делаем это максимально неэффективным, зато наглядным образом
        number = len(self.nodes[0])
        for i in range(0, number):
            for j in range(0, number):
                for k in range(0, number):
                    # Вставляем новую точку в сетку VTK-снапшота
                    points.InsertNextPoint(self.nodes[0][i,j,k], self.nodes[1][i,j,k], self.nodes[2][i,j,k])
                    # Добавляем значение скалярного поля в этой точке
                    smth.InsertNextValue(self.smth[i,j,k])
                    # Добавляем значение векторного поля в этой точке
                    vel.InsertNextTuple((self.velocity[0][i,j,k], self.velocity[1][i,j,k], self.velocity[2][i,j,k]))

        # Задаём размеры VTK-сетки (в точках, по трём осям)
        structuredGrid.SetDimensions(number, number, number)
        # Грузим точки в сетку
        structuredGrid.SetPoints(points)

        # Присоединяем векторное и скалярное поля к точкам
        structuredGrid.GetPointData().AddArray(smth)
        structuredGrid.GetPointData().AddArray(vel)

        # Создаём снапшот в файле с заданным именем
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetInputDataObject(structuredGrid)
        writer.SetFileName("cubic3d-step-" + str(snap_number) + ".vts")
        writer.Write()


# Размер расчётной сетки, точек на сторону
size = 10
# Шаг точек по пространству
h = 0.1
# Шаг по времени
tau = 0.01

# Создаём сетку заданного размера
m = CalcMesh(size, h)
# Пишем её начальное состояние в VTK
m.snapshot(0)

for i in range(1, 100):
    m.move(tau)
    m.snapshot(i)