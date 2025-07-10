import os
import json
import traceback
import pandas as pd
from typing import Annotated, Dict,  Optional
from reportlab.lib import colors
from reportlab.lib import pagesizes
from reportlab.platypus import (
    SimpleDocTemplate,
    Frame,
    Paragraph,
    Image,
    PageTemplate,
    FrameBreak,
    Spacer,
    Table,
    TableStyle,
    NextPageTemplate,
    PageBreak,
)
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT


class ReportLabUtils:
    @staticmethod
    def build_annual_report(
        ticker_symbol: str,
        filing_date: str,
        output_pdf_path: str,
        sections: dict,         # conceptual_summaries, e.g. {section_name: text}
        key_data: dict,         # loaded from key_data.json
        financial_metrics: dict,  # loaded from financial_metrics.json (parsed to DataFrame in code)
        chart_paths: dict,      # {'pe_eps_performance': ..., 'share_performance': ...}
        summaries: dict         # {filename: text} for appendix or extras
    ) -> str:
        """
        Builds the final PDF annual report from mapped text summaries, images, and tables.
        Uses LLM-generated asset_map for robust, adaptable section linking.
        Returns the PDF file path on success, or an error string on failure.
        """
        # Enforce asset_map format: dict of canonical keys to FILENAME (not path)
        REQUIRED_KEYS = [
            "company_overview",
            "key_financials",
            "valuation",
            "risk_assessment",
            "sell_side_summary",
            "competitors_analysis",
            "pe_eps_plot",
            "share_performance_plot",
        ]

        # --- IMAGE ASSETS ---
        share_performance_image_path = chart_paths["share_performance"]
        pe_eps_performance_image_path = chart_paths["pe_eps_performance"]

        pdf_path = output_pdf_path
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

        doc = SimpleDocTemplate(pdf_path, pagesize=pagesizes.A4)

        # 2. 创建PDF并插入图像
        # 页面设置
        page_width, page_height = pagesizes.A4
        left_column_width = page_width * 2 / 3
        right_column_width = page_width - left_column_width
        margin = 4
    
        # 定义两个栏位的Frame
        frame_left = Frame(
            margin,
            margin,
            left_column_width - margin * 2,
            page_height - margin * 2,
            id="left",
        )
        frame_right = Frame(
            left_column_width,
            margin,
            right_column_width - margin * 2,
            page_height - margin * 2,
            id="right",
        )

        single_frame = Frame(margin, margin, page_width-margin*2, page_height-margin*2, id='single')
        single_column_layout = PageTemplate(id='OneCol', frames=[single_frame])

        left_column_width_p2 = (page_width - margin * 3) // 2
        right_column_width_p2 = left_column_width_p2
        frame_left_p2 = Frame(
            margin,
            margin,
            left_column_width_p2 - margin * 2,
            page_height - margin * 2,
            id="left",
        )
        frame_right_p2 = Frame(
            left_column_width_p2,
            margin,
            right_column_width_p2 - margin * 2,
            page_height - margin * 2,
            id="right",
        )

        #创建PageTemplate，并添加到文档
        page_template = PageTemplate(
            id="TwoColumns", frames=[frame_left, frame_right]
        )
        page_template_p2 = PageTemplate(
            id="TwoColumns_p2", frames=[frame_left_p2, frame_right_p2]
        )

        #Define single column Frame
        single_frame = Frame(
            margin,
            margin,
            page_width - 2 * margin,
            page_height - 2 * margin,
            id="single",
        )

        # Create a PageTemplate with a single column
        single_column_layout = PageTemplate(id="OneCol", frames=[single_frame])

        doc.addPageTemplates([page_template, single_column_layout, page_template_p2])

        styles = getSampleStyleSheet()

        # 自定义样式
        custom_style = ParagraphStyle(
            name="Custom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            # leading=15,
            alignment=TA_JUSTIFY,
        )

        title_style = ParagraphStyle(
            name="TitleCustom",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            alignment=TA_LEFT,
            spaceAfter=10,
        )

        subtitle_style = ParagraphStyle(
            name="Subtitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=12,
            alignment=TA_LEFT,
            spaceAfter=6,
        )

        table_style2 = TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BACKGROUND", (0, 0), (-1, 0), colors.white),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 7),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                # 所有单元格左对齐
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                # 标题栏下方添加横线
                ("LINEBELOW", (0, 0), (-1, 0), 2, colors.black),
                # 表格最下方添加横线
                ("LINEBELOW", (0, -1), (-1, -1), 2, colors.black),
            ]
        )

        # df, currency, name = FMPUtils.get_financial_metrics(ticker_symbol, years=5) loaded at data inception
        currency = financial_metrics.get("currency", "USD")
        name = financial_metrics.get("company_name", "Unknown")

        metrics = financial_metrics["metrics"]
        df = pd.DataFrame(metrics).T
        df = df[sorted(df.columns, reverse=True)]
        
        def fmt_cell(x):
            # if it’s already a string (e.g. "2.0%") just leave it alone
            if isinstance(x, str):
                return x
            # if it’s missing
            if pd.isna(x):
                return ""
            # else force to float
            v = float(x)
            # if it’s really an integer value, show no decimals
            if v.is_integer():
                return f"{int(v):,}"
            # otherwise show two decimals
            return f"{v:,.2f}"

        df_formatted = df.copy()
        for col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(fmt_cell)
        # 准备左栏和右栏内容
        content = []
        # 标题
        content.append(
            Paragraph(
                f"Equity Research Report: {name}",
                title_style,
            )
        )

        # 子标题
        content.append(Paragraph("Business Overview", subtitle_style))
        content.append(Paragraph(summaries.get("company_overview.txt",""), custom_style))

        content.append(Paragraph("Market Position", subtitle_style))
        content.append(Paragraph(summaries.get("key_financials.txt",""), custom_style))
        
        content.append(Paragraph("Operating Results", subtitle_style))
        content.append(Paragraph(summaries.get("valuation.txt",""), custom_style))

        content.append(Paragraph("Summarization", subtitle_style))

        df_formatted.reset_index(inplace=True)
        df_formatted.rename(columns={"index": f"FY ({currency} mn)"}, inplace=True)

        # Transpose the table: metrics as rows, years as columns
        #df_flipped = df.set_index(f"FY ({currency} mn)")
        #df_flipped.reset_index(inplace=True)
        #df_flipped.rename(columns={"index": "Financial Metrics"}, inplace=True)
        #print("after currency", df)

        # Now use this transposed DataFrame for the table
        table_data = [df_formatted.columns.to_list()] + df_formatted.values.tolist()

        #table_data = [["Financial Metrics"]]
        #table_data += [df.columns.to_list()] + df.values.tolist()

        # Compute adaptive column widths based on column types
        base_width = (left_column_width - margin * 4)
        num_cols = len(df_formatted.columns)

        # Assign slightly larger width to potentially wider columns like "Gross Profit" or "Revenue"
        column_weights = [
            1.2 if "Profit" in col or "Revenue" in col else 1.0
            for col in df_formatted.columns
        ]
        total_weight = sum(column_weights)
        col_widths = [base_width * w / total_weight for w in column_weights]

        # Create the table
        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(table_style2)
        content.append(table)

        content.append(FrameBreak())  # 用于从左栏跳到右栏

        table_style = TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BACKGROUND", (0, 0), (-1, 0), colors.white),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 12),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                # 第一列左对齐
                ("ALIGN", (0, 1), (0, -1), "LEFT"),
                # 第二列右对齐
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                # 标题栏下方添加横线
                ("LINEBELOW", (0, 0), (-1, 0), 2, colors.black),
            ]
        )
        full_length = right_column_width - 2 * margin

        data = [
            ["FinRobot"],
            ["Team9 - UOA"],
            ["GitHub:ashreim-UPL/FinRobot-Langgraph.git"],
            [f"Report date: {filing_date}"],
        ]
        col_widths = [full_length]
        table = Table(data, colWidths=col_widths)
        table.setStyle(table_style)
        content.append(table)

        # content.append(Paragraph("", custom_style))
        content.append(Spacer(1, 0.15 * inch))
        # key_data = ReportAnalysisUtils.get_key_data(ticker_symbol, filing_date)  # removed is passed from data collection
        # 表格数据
        data = [["Key data", ""]]
        data += [[k, v] for k, v in key_data.items()]
        col_widths = [full_length // 3 * 2, full_length // 3]
        table = Table(data, colWidths=col_widths)
        table.setStyle(table_style)
        content.append(table)

        # 将Matplotlib图像添加到右栏

        # 历史股价
        data = [["Share Performance"]]
        col_widths = [full_length]
        table = Table(data, colWidths=col_widths)
        table.setStyle(table_style)
        content.append(table)

        plot_path = share_performance_image_path
        width = right_column_width
        height = width // 2
        if os.path.exists(plot_path):
            content.append(Image(plot_path, width=width, height=height))
        else:
            content.append(Paragraph(f"Image not found: {plot_path}", custom_style))

        # 历史PE和EPS
        data = [["PE & EPS"]]
        col_widths = [full_length]
        table = Table(data, colWidths=col_widths)
        table.setStyle(table_style)
        content.append(table)

        plot_path = pe_eps_performance_image_path
        width = right_column_width
        height = width // 2
        if os.path.exists(plot_path):
            content.append(Image(plot_path, width=width, height=height))
        else:
            content.append(Paragraph(f"Image not found: {plot_path}", custom_style))
        
        # # 开始新的一页
        content.append(NextPageTemplate("OneCol"))
        content.append(PageBreak())
        
        content.append(Paragraph("Risk Assessment", subtitle_style))
        content.append(Paragraph(summaries.get("risk_assessment.txt",""), custom_style))

        content.append(Paragraph("Competitors Analysis", subtitle_style))
        content.append(Paragraph(summaries.get("competitors_analysis.txt",""), custom_style))


        doc.build(content)

        return f"Success: Annual report generated successfully at {pdf_path}"

