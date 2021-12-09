from openpyxl import load_workbook, Workbook
from openpyxl.utils import get_column_letter


class ExcelWriter:
    def __init__(self, file):
        self.results_file = file

    def saveModel(self, i, agents, model_error, title):
        wb = load_workbook(self.results_file)
        ws = wb["Training_Results"]
        col = i * 2
        row = 1
        ws.cell(row, col, title + str(i) + " (MAE)")
        ws.cell(row, col + 1, title + str(i) + " (%)")
        row_n = 2
        row_c = 3
        for a in agents:
            agent = agents[a]
            # print("Error for :", a)
            # print("Naive-->" + str(agent.n_error) + ":" + str(agent.p_error[0]))
            # print("Collab-->" + str(agent.error) + ":" + str(agent.p_error[1]))
            ws.cell(row_n, col, agent.n_error)
            ws.cell(row_n, col + 1, agent.p_error[0])
            ws.cell(row_c, col, agent.error)
            ws.cell(row_c, col + 1, agent.p_error[1])
            row_n += 2
            row_c += 2
        ws.cell(row_n, col, model_error['MAE']['average'])
        ws.cell(row_n, col + 1, model_error['p']['average'])
        ws.cell(row_n + 1, col, model_error['MAE']['error_w'])
        ws.cell(row_n + 1, col + 1, model_error['p']['error_w'])
        ws.cell(row_n + 2, col, model_error['MAE']['trust_w'])
        ws.cell(row_n + 2, col + 1, model_error['p']['trust_w'])
        wb.save(self.results_file)

    def initializeFile(self, agents):
        wb = Workbook()
        ws = wb.active
        ws.title = "Training_Results"
        ws.cell(1, 1, "Agents")
        row_n = 2
        row_c = 3
        for s in agents:
            ws.cell(row_n, 1, "N" + str(s))
            ws.cell(row_c, 1, s)
            row_c += 2
            row_n += 2
        ws.cell(row_n, 1, "Model AVG")
        ws.cell(row_n + 1, 1, "Model ERR")
        ws.cell(row_n + 2, 1, "Model TF")
        wb.save(filename=self.results_file)

    def saveIter(self, values, collab_predictions, naive_predictions, model_vals, i, intervals, agents):
        wb = load_workbook(self.results_file)
        ws = wb.create_sheet("Iteration #"+str(i))
        col = 1
        row = 1
        ws.cell(row, col, "Values")
        ws.column_dimensions[get_column_letter(col)].width = 20
        # write time interval col headings
        for i in intervals:
            col += 1
            ws.cell(row, col, i)
            ws.column_dimensions[get_column_letter(col)].width = 20
        col += 1
        ws.cell(row, col, "Agent Bias")
        # write actual values, and 3x model aggregation values
        actual_ind = 2
        modelavg_ind = 3
        error_ind = 4
        tf_ind = 5
        num_i = len(intervals)
        ws.cell(actual_ind, 1, "Real Values")
        ws.cell(modelavg_ind, 1, "Model AVG")
        ws.cell(error_ind, 1, "Model ERR")
        ws.cell(tf_ind, 1, "Model TF")
        for i in range(2, num_i + 2):
            ws.cell(actual_ind, i, values[i - 2])
            ws.cell(modelavg_ind, i, model_vals[i - 2]['average'])
            ws.cell(error_ind, i, model_vals[i - 2]['error'])
            ws.cell(tf_ind, i, model_vals[i - 2]['trust'])

        # write naive and collab values from each agent
        agent_n_index = 6
        agent_c_index = 7
        for a in collab_predictions:
            col = 1
            ws.cell(agent_n_index, col, "Agent #" + str(a) + ":Naive")
            ws.cell(agent_c_index, col, "Agent #" + str(a) + ":Collab")
            num = len(collab_predictions[a])
            for p in range(0, num):
                col += 1
                ws.cell(agent_n_index, col, naive_predictions[a][p])
                ws.cell(agent_c_index, col, collab_predictions[a][p])
            col += 1
            ws.cell(agent_c_index, col, agents[a].bias)
            agent_n_index += 2
            agent_c_index += 2
        wb.save(self.results_file)
