#define N_DATA 7200

int ecg_data[N_DATA] = {-29,
-29,
-29,
-29,
-24,
-27,
-31,
-31,
-35,
-37,
-34,
-35,
-37,
-30,
-32,
-38,
-46,
-49,
-53,
-55,
-55,
-52,
-56,
-58,
-58,
-57,
-58,
-55,
-56,
-61,
-58,
-57,
-61,
-65,
-73,
-77,
-89,
-97,
-78,
-38,
-14,
114,
163,
134,
21,
-33,
-86,
-70,
-65,
-69,
-66,
-63,
-65,
-67,
-69,
-67,
-67,
-65,
-67,
-71,
-67,
-63,
-65,
-68,
-68,
-69,
-69,
-66,
-66,
-69,
-65,
-66,
-67,
-68,
-68,
-70,
-70,
-65,
-67,
-69,
-66,
-68,
-70,
-68,
-67,
-66,
-70,
-67,
-68,
-71,
-71,
-68,
-71,
-72,
-73,
-72,
-75,
-72,
-73,
-75,
-73,
-72,
-73,
-69,
-65,
-65,
-64,
-60,
-58,
-58,
-57,
-51,
-51,
-51,
-46,
-49,
-51,
-49,
-51,
-51,
-50,
-53,
-52,
-54,
-52,
-55,
-56,
-53,
-56,
-57,
-55,
-53,
-57,
-56,
-57,
-56,
-60,
-59,
-62,
-61,
-59,
-57,
-60,
-58,
-60,
-61,
-63,
-59,
-63,
-66,
-63,
-61,
-65,
-65,
-62,
-61,
-61,
-60,
-60,
-61,
-61,
-58,
-62,
-61,
-58,
-56,
-55,
-52,
-50,
-45,
-44,
-40,
-44,
-44,
-45,
-45,
-47,
-46,
-42,
-50,
-57,
-61,
-66,
-65,
-64,
-63,
-68,
-68,
-70,
-70,
-66,
-65,
-70,
-68,
-65,
-66,
-73,
-81,
-91,
-103,
-107,
-65,
-15,
67,
150,
175,
157,
59,
-34,
-80,
-90,
-87,
-78,
-77,
-76,
-76,
-78,
-77,
-76,
-82,
-82,
-79,
-80,
-77,
-78,
-78,
-82,
-78,
-79,
-83,
-80,
-81,
-83,
-80,
-79,
-78,
-79,
-78,
-79,
-81,
-79,
-77,
-80,
-80,
-78,
-77,
-83,
-80,
-81,
-84,
-84,
-84,
-85,
-86,
-84,
-86,
-91,
-88,
-90,
-94,
-92,
-90,
-94,
-92,
-89,
-88,
-84,
-76,
-73,
-72,
-68,
-63,
-66,
-64,
-60,
-60,
-64,
-61,
-61,
-65,
-61,
-60,
-64,
-62,
-63,
-61,
-63,
-59,
-61,
-64,
-63,
-63,
-64,
-63,
-63,
-65,
-68,
-67,
-66,
-70,
-69,
-66,
-69,
-67,
-66,
-67,
-68,
-66,
-68,
-72,
-71,
-68,
-69,
-68,
-66,
-67,
-70,
-67,
-69,
-71,
-70,
-67,
-68,
-66,
-66,
-67,
-69,
-65,
-59,
-59,
-57,
-53,
-52,
-50,
-45,
-43,
-49,
-48,
-49,
-55,
-55,
-46,
-45,
-53,
-56,
-57,
-67,
-66,
-68,
-74,
-72,
-70,
-74,
-76,
-73,
-73,
-77,
-73,
-75,
-81,
-82,
-97,
-108,
-111,
-75,
-50,
70,
161,
184,
86,
10,
-100,
-86,
-79,
-79,
-78,
-80,
-80,
-79,
-82,
-81,
-82,
-82,
-83,
-81,
-84,
-85,
-82,
-82,
-86,
-84,
-82,
-85,
-80,
-80,
-80,
-82,
-79,
-81,
-85,
-85,
-83,
-85,
-84,
-80,
-80,
-83,
-80,
-80,
-81,
-77,
-78,
-81,
-81,
-82,
-83,
-84,
-83,
-84,
-86,
-87,
-84,
-84,
-85,
-79,
-79,
-73,
-68,
-66,
-67,
-65,
-61,
-63,
-61,
-60,
-60,
-62,
-62,
-61,
-64,
-65,
-61,
-64,
-61,
-59,
-62,
-65,
-64,
-67,
-70,
-68,
-66,
-70,
-70,
-71,
-72,
-72,
-69,
-71,
-73,
-73,
-71,
-75,
-74,
-73,
-72,
-75,
-72,
-70,
-74,
-73,
-69,
-72,
-72,
-71,
-71,
-73,
-72,
-72,
-74,
-72,
-70,
-69,
-66,
-61,
-61,
-59,
-56,
-56,
-55,
-53,
-51,
-54,
-55,
-54,
-55,
-55,
-54,
-50,
-52,
-56,
-67,
-74,
-77,
-74,
-75,
-78,
-77,
-77,
-81,
-80,
-77,
-81,
-81,
-78,
-81,
-94,
-101,
-106,
-121,
-129,
-103,
-64,
-7,
84,
127,
158,
46,
-76,
-103,
-95,
-78,
-81,
-80,
-78,
-79,
-82,
-79,
-80,
-83,
-82,
-81,
-81,
-81,
-77,
-78,
-80,
-80,
-79,
-81,
-80,
-76,
-78,
-78,
-75,
-77,
-78,
-75,
-77,
-78,
-78,
-76,
-78,
-77,
-76,
-78,
-75,
-75,
-76,
-77,
-73,
-76,
-78,
-75,
-77,
-76,
-75,
-73,
-77,
-77,
-77,
-73,
-72,
-66,
-63,
-64,
-60,
-58,
-59,
-57,
-56,
-57,
-59,
-58,
-58,
-59,
-58,
-56,
-59,
-60,
-59,
-59,
-61,
-59,
-61,
-63,
-60,
-61,
-66,
-66,
-64,
-64,
-67,
-63,
-64,
-66,
-68,
-66,
-69,
-69,
-68,
-67,
-67,
-66,
-68,
-68,
-73,
-69,
-70,
-70,
-68,
-69,
-74,
-71,
-69,
-70,
-67,
-66,
-67,
-66,
-64,
-59,
-59,
-56,
-56,
-55,
-51,
-47,
-49,
-51,
-49,
-52,
-52,
-54,
-55,
-51,
-52,
-58,
-63,
-67,
-69,
-68,
-71,
-68,
-69,
-72,
-75,
-73,
-76,
-75,
-73,
-73,
-75,
-72,
-71,
-74,
-84,
-93,
-107,
-108,
-95,
-18,
71,
150,
147,
107,
-75,
-98,
-87,
-80,
-78,
-80,
-85,
-81,
-81,
-81,
-78,
-78,
-79,
-79,
-77,
-79,
-84,
-80,
-80,
-81,
-82,
-81,
-82,
-81,
-84,
-82,
-84,
-81,
-79,
-79,
-79,
-81,
-85,
-83,
-81,
-82,
-81,
-80,
-79,
-81,
-81,
-79,
-82,
-85,
-85,
-89,
-89,
-87,
-94,
-91,
-88,
-85,
-89,
-89,
-86,
-84,
-83,
-76,
-73,
-73,
-67,
-62,
-62,
-62,
-62,
-62,
-64,
-61,
-59,
-62,
-62,
-59,
-61,
-57,
-56,
-60,
-63,
-60,
-63,
-65,
-64,
-65,
-66,
-65,
-65,
-66,
-68,
-67,
-66,
-63,
-66,
-66,
-67,
-66,
-64,
-66,
-70,
-68,
-64,
-65,
-64,
-64,
-66,
-67,
-66,
-67,
-68,
-65,
-65,
-69,
-67,
-63,
-61,
-55,
-54,
-52,
-51,
-47,
-44,
-47,
-45,
-46,
-49,
-48,
-46,
-50,
-48,
-41,
-42,
-46,
-56,
-60,
-66,
-68,
-69,
-69,
-70,
-68,
-69,
-71,
-73,
-69,
-73,
-72,
-69,
-70,
-77,
-81,
-91,
-95,
-109,
-89,
-49,
19,
63,
173,
126,
-1,
-88,
-100,
-82,
-78,
-79,
-78,
-76,
-79,
-82,
-78,
-81,
-81,
-80,
-79,
-83,
-83,
-80,
-81,
-82,
-79,
-79,
-81,
-81,
-78,
-83,
-83,
-81,
-82,
-85,
-81,
-78,
-80,
-81,
-80,
-84,
-83,
-83,
-84,
-84,
-80,
-80,
-80,
-80,
-80,
-83,
-85,
-82,
-85,
-89,
-87,
-86,
-90,
-88,
-88,
-88,
-86,
-86,
-80,
-75,
-67,
-65,
-66,
-60,
-57,
-59,
-59,
-57,
-59,
-59,
-56,
-55,
-58,
-55,
-54,
-56,
-57,
-55,
-58,
-60,
-59,
-58,
-60,
-60,
-60,
-63,
-65,
-63,
-63,
-65,
-62,
-63,
-66,
-63,
-62,
-66,
-66,
-64,
-65,
-67,
-65,
-64,
-65,
-64,
-63,
-64,
-66,
-62,
-66,
-67,
-66,
-64,
-66,
-64,
-63,
-66,
-64,
-63,
-60,
-62,
-54,
-51,
-52,
-48,
-46,
-44,
-43,
-40,
-44,
-47,
-45,
-48,
-48,
-43,
-42,
-52,
-57,
-60,
-65,
-65,
-66,
-67,
-68,
-68,
-64,
-67,
-69,
-69,
-68,
-70,
-68,
-68,
-69,
-73,
-85,
-97,
-104,
-107,
-62,
-3,
82,
167,
189,
103,
-33,
-91,
-84,
-76,
-73,
-76,
-75,
-73,
-74,
-76,
-74,
-77,
-77,
-77,
-78,
-81,
-78,
-79,
-79,
-80,
-78,
-79,
-78,
-77,
-75,
-78,
-74,
-74,
-75,
-77,
-75,
-76,
-78,
-77,
-76,
-79,
-77,
-76,
-77,
-76,
-74,
-75,
-73,
-73,
-75,
-79,
-75,
-74,
-77,
-78,
-77,
-77,
-76,
-74,
-77,
-78,
-74,
-75,
-77,
-73,
-69,
-66,
-63,
-62,
-57,
-58,
-54,
-56,
-56,
-57,
-55,
-57,
-58,
-56,
-57,
-60,
-57,
-57,
-57,
-57,
-55,
-59,
-61,
-57,
-54,
-57,
-54,
-53,
-53,
-50,
-47,
-50,
-48,
-43,
-50,
-43,
-42,
-46,
-48,
-49,
-55,
-63,
-64,
-62,
-68,
-72,
-72,
-71,
-75,
-75,
-73,
-74,
-72,
-70,
-71,
-72,
-70,
-71,
-70,
-74,
-83,
-98,
-105,
-110,
-93,
-42,
23,
111,
147,
153,
29,
-78,
-92,
-83,
-76,
-78,
-77,
-77,
-79,
-77,
-77,
-79,
-81,
-80,
-78,
-79,
-77,
-78,
-82,
-81,
-78,
-78,
-80,
-78,
-78,
-79,
-80,
-79,
-82,
-78,
-77,
-80,
-79,
-78,
-78,
-78,
-76,
-77,
-78,
-78,
-77,
-78,
-79,
-77,
-78,
-81,
-78,
-76,
-77,
-79,
-79,
-82,
-84,
-81,
-81,
-82,
-78,
-77,
-79,
-75,
-74,
-76,
-71,
-69,
-67,
-69,
-64,
-61,
-61,
-63,
-61,
-66,
-69,
-65,
-62,
-62,
-60,
-61,
-63,
-64,
-62,
-65,
-66,
-63,
-65,
-68,
-67,
-71,
-73,
-69,
-68,
-72,
-72,
-72,
-72,
-71,
-70,
-69,
-71,
-70,
-70,
-70,
-72,
-69,
-71,
-73,
-71,
-71,
-72,
-73,
-73,
-75,
-75,
-74,
-68,
-72,
-70,
-68,
-72,
-73,
-72,
-75,
-77,
-73,
-75,
-74,
-71,
-72,
-73,
-71,
-68,
-68,
-69,
-70,
-71,
-73,
-72,
-72,
-75,
-71,
-71,
-73,
-74,
-73,
-71,
-71,
-68,
-69,
-71,
-72,
-70,
-70,
-71,
-72,
-67,
-64,
-61,
-61,
-59,
-56,
-52,
-51,
-53,
-51,
-55,
-57,
-53,
-55,
-58,
-60,
-54,
-55,
-60,
-62,
-68,
-74,
-73,
-74,
-74,
-73,
-75,
-81,
-81,
-80,
-79,
-82,
-79,
-79,
-82,
-90,
-98,
-108,
-113,
-105,
-43,
24,
115,
170,
177,
59,
-39,
-91,
-110,
-107,
-91,
-89,
-84,
-82,
-83,
-85,
-86,
-88,
-90,
-86,
-88,
-89,
-85,
-86,
-85,
-87,
-87,
-89,
-87,
-86,
-86,
-86,
-86,
-86,
-88,
-86,
-84,
-88,
-89,
-89,
-87,
-88,
-85,
-83,
-85,
-82,
-82,
-83,
-85,
-81,
-81,
-85,
-82,
-86,
-89,
-91,
-91,
-94,
-99,
-95,
-100,
-102,
-97,
-97,
-100,
-97,
-92,
-94,
-88,
-86,
-82,
-79,
-74,
-70,
-71,
-67,
-63,
-64,
-61,
-60,
-61,
-64,
-63,
-64,
-65,
-63,
-62,
-65,
-65,
-65,
-65,
-67,
-63,
-64,
-67,
-65,
-64,
-67,
-69,
-69,
-70,
-71,
-69,
-67,
-69,
-68,
-66,
-69,
-70,
-68,
-70,
-70,
-67,
-70,
-72,
-70,
-69,
-71,
-72,
-69,
-71,
-71,
-70,
-69,
-71,
-69,
-68,
-73,
-70,
-66,
-60,
-60,
-59,
-55,
-52,
-53,
-50,
-51,
-52,
-52,
-55,
-56,
-57,
-51,
-49,
-56,
-59,
-65,
-71,
-70,
-74,
-77,
-72,
-72,
-73,
-74,
-72,
-75,
-77,
-73,
-76,
-79,
-77,
-80,
-85,
-102,
-109,
-113,
-76,
-51,
61,
145,
174,
101,
37,
-102,
-96,
-86,
-83,
-83,
-80,
-86,
-83,
-83,
-83,
-83,
-82,
-82,
-80,
-79,
-80,
-80,
-79,
-80,
-80,
-81,
-80,
-81,
-83,
-81,
-80,
-82,
-83,
-81,
-81,
-81,
-80,
-81,
-79,
-78,
-78,
-84,
-81,
-80,
-83,
-80,
-80,
-84,
-83,
-83,
-83,
-86,
-82,
-81,
-83,
-85,
-82,
-84,
-85,
-84,
-83,
-79,
-74,
-70,
-70,
-65,
-62,
-63,
-61,
-61,
-58,
-58,
-56,
-56,
-56,
-56,
-56,
-59,
-59,
-57,
-59,
-61,
-60,
-57,
-61,
-60,
-61,
-62,
-64,
-62,
-63,
-67,
-62,
-65,
-65,
-65,
-65,
-68,
-68,
-64,
-67,
-69,
-66,
-68,
-70,
-68,
-67,
-68,
-67,
-67,
-66,
-68,
-64,
-64,
-67,
-64,
-63,
-66,
-67,
-64,
-64,
-67,
-64,
-62,
-62,
-52,
-50,
-51,
-48,
-44,
-44,
-44,
-46,
-47,
-49,
-48,
-48,
-46,
-40,
-44,
-53,
-59,
-61,
-64,
-66,
-68,
-67,
-68,
-70,
-69,
-70,
-72,
-69,
-67,
-67,
-70,
-75,
-92,
-101,
-105,
-84,
-29,
51,
144,
175,
107,
-37,
-95,
-81,
-73,
-75,
-77,
-74,
-75,
-77,
-77,
-76,
-79,
-81,
-78,
-77,
-81,
-78,
-77,
-79,
-77,
-75,
-78,
-77,
-76,
-78,
-78,
-75,
-73,
-77,
-77,
-74,
-74,
-75,
-74,
-75,
-75,
-75,
-74,
-75,
-76,
-75,
-75,
-77,
-77,
-75,
-76,
-76,
-75,
-78,
-77,
-75,
-77,
-78,
-74,
-74,
-73,
-68,
-65,
-66,
-64,
-63,
-60,
-57,
-54,
-54,
-55,
-51,
-54,
-56,
-57,
-54,
-55,
-56,
-55,
-54,
-58,
-57,
-54,
-56,
-55,
-54,
-58,
-59,
-58,
-59,
-60,
-58,
-58,
-62,
-63,
-60,
-64,
-64,
-61,
-62,
-64,
-59,
-61,
-62,
-63,
-61,
-65,
-64,
-62,
-64,
-63,
-60,
-57,
-58,
-61,
-61,
-63,
-62,
-59,
-62,
-60,
-51,
-51,
-48,
-45,
-43,
-40,
-39,
-36,
-40,
-43,
-42,
-42,
-43,
-38,
-35,
-49,
-51,
-54,
-59,
-61,
-61,
-59,
-60,
-62,
-60,
-61,
-65,
-64,
-64,
-68,
-65,
-65,
-69,
-83,
-91,
-103,
-93,
-74,
13,
108,
167,
136,
90,
-79,
-87,
-76,
-71,
-71,
-70,
-71,
-68,
-67,
-67,
-69,
-67,
-71,
-72,
-73,
-72,
-71,
-71,
-71,
-70,
-67,
-67,
-70,
-69,
-67,
-71,
-73,
-70,
-73,
-73,
-72,
-69,
-69,
-67,
-65,
-69,
-70,
-71,
-70,
-72,
-69,
-68,
-69,
-70,
-68,
-72,
-76,
-71,
-71,
-74,
-74,
-70,
-73,
-72,
-69,
-66,
-65,
-60,
-58,
-58,
-56,
-52,
-54,
-51,
-50,
-50,
-53,
-50,
-50,
-51,
-54,
-54,
-56,
-56,
-55,
-55,
-57,
-57,
-58,
-56,
-56,
-55,
-59,
-60,
-60,
-63,
-62,
-60,
-62,
-64,
-63,
-62,
-63,
-63,
-61,
-63,
-65,
-63,
-61,
-60,
-61,
-62,
-62,
-62,
-60,
-62,
-65,
-61,
-59,
-63,
-60,
-54,
-52,
-52,
-50,
-44,
-46,
-41,
-42,
-42,
-47,
-46,
-51,
-52,
-49,
-47,
-56,
-56,
-62,
-65,
-66,
-63,
-64,
-66,
-65,
-68,
-70,
-70,
-70,
-73,
-70,
-69,
-71,
-77,
-81,
-92,
-106,
-90,
-47,
-23,
112,
162,
120,
-5,
-51,
-94,
-83,
-75,
-75,
-75,
-77,
-77,
-77,
-74,
-74,
-76,
-76,
-74,
-75,
-77,
-78,
-78,
-78,
-81,
-78,
-80,
-81,
-75,
-78,
-79,
-78,
-79,
-80,
-79,
-78,
-79,
-81,
-78,
-79,
-80,
-77,
-75,
-77,
-78,
-76,
-77,
-79,
-80,
-83,
-84,
-86,
-85,
-90,
-91,
-88,
-89,
-88,
-85,
-85,
-86,
-83,
-76,
-76,
-72,
-70,
-66,
-64,
-61,
-60,
-58,
-58,
-57,
-57,
-58,
-57,
-55,
-58,
-55,
-55,
-58,
-61,
-57,
-61,
-62,
-59,
-60,
-61,
-60,
-61,
-61,
-62,
-59,
-61,
-61,
-61,
-62,
-65,
-64,
-62,
-65,
-65,
-63,
-66,
-65,
-62,
-63,
-66,
-60,
-61,
-63,
-66,
-64,
-65,
-65,
-63,
-63,
-64,
-62,
-62,
-63,
-63,
-59,
-61,
-60,
-56,
-54,
-54,
-48,
-47,
-48,
-44,
-41,
-42,
-44,
-44,
-44,
-46,
-46,
-44,
-46,
-43,
-39,
-47,
-55,
-56,
-63,
-65,
-65,
-64,
-65,
-65,
-62,
-64,
-66,
-64,
-66,
-72,
-66,
-65,
-67,
-68,
-74,
-87,
-94,
-97,
-94,
-46,
21,
112,
148,
171,
57,
-61,
-90,
-84,
-72,
-75,
-71,
-74,
-75,
-74,
-74,
-74,
-74,
-75,
-75,
-74,
-72,
-71,
-71,
-72,
-70,
-70,
-72,
-72,
-72,
-73,
-70,
-70,
-71,
-73,
-70,
-71,
-74,
-71,
-71,
-74,
-72,
-71,
-73,
-74,
-70,
-72,
-71,
-69,
-70,
-74,
-71,
-73,
-73,
-76,
-74,
-77,
-75,
-77,
-76,
-77,
-74,
-70,
-67,
-60,
-58,
-57,
-56,
-56,
-53,
-54,
-50,
-50,
-52,
-48,
-49,
-51,
-50,
-50,
-50,
-53,
-48,
-49,
-50,
-53,
-53,
-55,
-55,
-53,
-55,
-57,
-56,
-56,
-57,
-58,
-55,
-59,
-58,
-59,
-60,
-63,
-61,
-59,
-62,
-63,
-60,
-63,
-63,
-64,
-60,
-63,
-60,
-58,
-62,
-62,
-61,
-63,
-64,
-62,
-62,
-63,
-60,
-59,
-59,
-59,
-58,
-60,
-60,
-57,
-55,
-51,
-46,
-48,
-48,
-44,
-42,
-40,
-41,
-39,
-42,
-45,
-40,
-42,
-45,
-37,
-38,
-50,
-55,
-56,
-61,
-65,
-64,
-65,
-66,
-67,
-65,
-68,
-68,
-66,
-67,
-68,
-67,
-67,
-69,
-84,
-89,
-92,
-97,
-102,
-111,
-85,
-40,
21,
57,
167,
159,
53,
-63,
-87,
-79,
-75,
-74,
-72,
-72,
-74,
-72,
-73,
-76,
-72,
-74,
-76,
-71,
-72,
-73,
-72,
-70,
-74,
-73,
-71,
-76,
-77,
-71,
-72,
-71,
-72,
-69,
-73,
-75,
-71,
-73,
-73,
-71,
-73,
-77,
-72,
-70,
-74,
-76,
-75,
-72,
-74,
-71,
-70,
-72,
-74,
-75,
-78,
-76,
-74,
-73,
-77,
-73,
-73,
-72,
-69,
-65,
-63,
-62,
-59,
-58,
-55,
-53,
-52,
-55,
-53,
-52,
-54,
-55,
-51,
-50,
-52,
-51,
-53,
-56,
-56,
-56,
-56,
-57,
-56,
-59,
-61,
-58,
-60,
-58,
-60,
-58,
-61,
-63,
-62,
-62,
-65,
-64,
-64,
-65,
-66,
-62,
-64,
-64,
-62,
-63,
-66,
-65,
-62,
-64,
-64,
-64,
-67,
-68,
-66,
-66,
-67,
-62,
-63,
-66,
-61,
-57,
-53,
-53,
-52,
-51,
-50,
-44,
-42,
-45,
-45,
-46,
-48,
-46,
-47,
-50,
-46,
-41,
-49,
-53,
-55,
-56,
-65,
-68,
-66,
-69,
-71,
-70,
-70,
-71,
-72,
-69,
-72,
-74,
-71,
-73,
-80,
-89,
-100,
-108,
-100,
-60,
-8,
81,
120,
144,
47,
-51,
-94,
-98,
-86,
-82,
-82,
-80,
-79,
-82,
-81,
-79,
-79,
-79,
-80,
-81,
-81,
-82,
-80,
-79,
-80,
-78,
-78,
-82,
-82,
-82,
-84,
-84,
-83,
-84,
-85,
-85,
-82,
-82,
-82,
-80,
-82,
-84,
-82,
-82,
-83,
-82,
-82,
-83,
-81,
-79,
-83,
-83,
-83,
-85,
-89,
-88,
-87,
-88,
-87,
-87,
-89,
-88,
-87,
-83,
-78,
-75,
-74,
-74,
-67,
-61,
-64,
-65,
-63,
-63,
-66,
-62,
-62,
-61,
-62,
-60,
-62,
-64,
-63,
-65,
-67,
-64,
-63,
-65,
-65,
-60,
-64,
-66,
-64,
-67,
-70,
-70,
-69,
-72,
-69,
-69,
-70,
-71,
-68,
-69,
-71,
-71,
-69,
-70,
-73,
-70,
-72,
-72,
-70,
-70,
-69,
-68,
-68,
-69,
-71,
-68,
-70,
-71,
-68,
-67,
-71,
-65,
-63,
-63,
-60,
-55,
-56,
-52,
-47,
-48,
-51,
-49,
-51,
-53,
-57,
-54,
-51,
-55,
-59,
-65,
-71,
-70,
-71,
-74,
-72,
-72,
-75,
-76,
-75,
-76,
-78,
-74,
-74,
-76,
-76,
-75,
-84,
-93,
-96,
-109,
-97,
-54,
3,
37,
156,
158,
67,
-49,
-78,
-95,
-87,
-83,
-83,
-83,
-80,
-78,
-81,
-82,
-80,
-81,
-86,
-82,
-83,
-86,
-84,
-85,
-84,
-85,
-84,
-84,
-85,
-82,
-82,
-84,
-84,
-80,
-84,
-84,
-84,
-84,
-87,
-85,
-84,
-85,
-86,
-85,
-83,
-86,
-85,
-81,
-84,
-84,
-84,
-87,
-85,
-88,
-91,
-93,
-93,
-93,
-98,
-92,
-92,
-91,
-86,
-82,
-81,
-77,
-73,
-73,
-73,
-68,
-66,
-66,
-65,
-62,
-64,
-64,
-63,
-62,
-65,
-64,
-64,
-66,
-66,
-65,
-68,
-68,
-64,
-67,
-69,
-68,
-67,
-69,
-69,
-66,
-70,
-74,
-70,
-71,
-70,
-67,
-64,
-67,
-70,
-68,
-71,
-72,
-71,
-72,
-72,
-70,
-71,
-71,
-70,
-69,
-71,
-71,
-70,
-70,
-69,
-66,
-69,
-70,
-68,
-66,
-66,
-61,
-57,
-56,
-57,
-51,
-48,
-48,
-46,
-45,
-49,
-51,
-48,
-53,
-51,
-46,
-54,
-59,
-64,
-65,
-70,
-73,
-71,
-68,
-72,
-69,
-71,
-70,
-74,
-72,
-74,
-73,
-74,
-75,
-77,
-82,
-93,
-97,
-107,
-72,
-18,
63,
107,
180,
100,
-29,
-91,
-95,
-77,
-76,
-79,
-81,
-79,
-78,
-81,
-79,
-77,
-81,
-78,
-76,
-81,
-80,
-79,
-81,
-81,
-79,
-79,
-79,
-81,
-78,
-80,
-77,
-77,
-75,
-78,
-78,
-76,
-78,
-79,
-74,
-78,
-80,
-78,
-76,
-78,
-74,
-73,
-75,
-75,
-73,
-77,
-79,
-81,
-79,
-81,
-78,
-76,
-77,
-76,
-74,
-72,
-70,
-64,
-62,
-60,
-56,
-54,
-56,
-56,
-53,
-56,
-58,
-55,
-54,
-58,
-55,
-52,
-55,
-54,
-55,
-58,
-60,
-56,
-57,
-60,
-57,
-57,
-59,
-60,
-57,
-59,
-61,
-60,
-61,
-62,
-61,
-60,
-64,
-64,
-62,
-63,
-64,
-62,
-62,
-63,
-61,
-60,
-62,
-62,
-59,
-65,
-66,
-64,
-62,
-65,
-62,
-60,
-59,
-61,
-55,
-52,
-47,
-46,
-46,
-47,
-44,
-41,
-40,
-39,
-39,
-43,
-45,
-44,
-45,
-50,
-44,
-37,
-39,
-50,
-54,
-62,
-65,
-65,
-66,
-67,
-66,
-67,
-69,
-70,
-70,
-69,
-70,
-68,
-70,
-84,
-92,
-101,
-108,
-76,
-21,
57,
140,
164,
77,
-46,
-89,
-81,
-76,
-75,
-73,
-74,
-74,
-72,
-72,
-76,
-74,
-73,
-76,
-77,
-75,
-76,
-76,
-75,
-74,
-75,
-72,
-71,
-71,
-71,
-72,
-73,
-75,
-71,
-74,
-76,
-74,
-72,
-74,
-72,
-70,
-71,
-72,
-69,
-70,
-71,
-68,
-70,
-71,
-72,
-73,
-74,
-76,
-73,
-72,
-76,
-74,
-73,
-75,
-73,
-71,
-71,
-70,
-68,
-66,
-67,
-62,
-59,
-61,
-57,
-53,
-57,
-55,
-54,
-53,
-55,
-52,
-52,
-54,
-58,
-56,
-59,
-60,
-60,
-59,
-61,
-61,
-59,
-60,
-63,
-61,
-61,
-61,
-61,
-60,
-62,
-63,
-64,
-67,
-63,
-65,
-66,
-65,
-64,
-64,
-68,
-65,
-66,
-67,
-66,
-64,
-67,
-67,
-65,
-64,
-66,
-64,
-62,
-64,
-66,
-61,
-58,
-54,
-53,
-52,
-54,
-51,
-47,
-46,
-47,
-47,
-48,
-48,
-44,
-48,
-50,
-43,
-52,
-57,
-65,
-68,
-70,
-71,
-68,
-69,
-72,
-69,
-70,
-70,
-70,
-70,
-72,
-75,
-78,
-94,
-104,
-110,
-80,
-67,
-7,
72,
147,
186,
194,
94,
-9,
-76,
-93,
-91,
-80,
-75,
-79,
-80,
-82,
-81,
-80,
-78,
-77,
-81,
-81,
-77,
-79,
-80,
-78,
-81,
-81,
-77,
-81,
-84,
-82,
-81,
-82,
-82,
-80,
-78,
-79,
-78,
-80,
-84,
-83,
-79,
-81,
-83,
-80,
-77,
-81,
-78,
-75,
-78,
-84,
-83,
-89,
-91,
-93,
-90,
-94,
-91,
-90,
-91,
-89,
-84,
-84,
-81,
-74,
-70,
-68,
-63,
-64,
-63,
-61,
-60,
-63,
-62,
-61,
-62,
-64,
-61,
-61,
-62,
-61,
-61,
-62,
-63,
-66,
-67,
-68,
-66,
-64,
-65,
-68,
-68,
-70,
-72,
-69,
-67,
-68,
-67,
-67,
-67,
-70,
-69,
-71,
-72,
-71,
-69,
-71,
-68,
-68,
-69,
-73,
-69,
-73,
-70,
-68,
-71,
-70,
-67,
-68,
-66,
-61,
-57,
-56,
-55,
-54,
-49,
-50,
-45,
-48,
-51,
-53,
-50,
-54,
-54,
-50,
-48,
-55,
-56,
-63,
-67,
-71,
-72,
-75,
-72,
-71,
-73,
-73,
-71,
-69,
-71,
-72,
-71,
-73,
-74,
-78,
-97,
-109,
-107,
-67,
-44,
83,
165,
165,
54,
-11,
-97,
-87,
-81,
-81,
-84,
-81,
-78,
-79,
-79,
-80,
-79,
-82,
-78,
-77,
-81,
-79,
-76,
-81,
-82,
-80,
-80,
-80,
-76,
-77,
-78,
-80,
-78,
-82,
-81,
-80,
-81,
-83,
-81,
-81,
-80,
-80,
-76,
-79,
-79,
-78,
-80,
-81,
-80,
-82,
-83,
-84,
-84,
-88,
-87,
-87,
-88,
-86,
-84,
-81,
-80,
-75,
-69,
-71,
-67,
-63,
-60,
-62,
-59,
-58,
-58,
-59,
-56,
-58,
-57,
-56,
-55,
-58,
-56,
-58,
-59,
-58,
-55,
-60,
-62,
-60,
-59,
-62,
-60,
-62,
-63,
-62,
-59,
-62,
-63,
-63,
-62,
-66,
-65,
-62,
-64,
-63,
-61,
-64,
-63,
-62,
-64,
-64,
-64,
-62,
-62,
-58,
-60,
-62,
-60,
-63,
-64,
-64,
-60,
-62,
-63,
-63,
-58,
-53,
-49,
-46,
-46,
-45,
-40,
-38,
-38,
-41,
-42,
-45,
-45,
-44,
-48,
-43,
-39,
-45,
-49,
-54,
-58,
-60,
-62,
-64,
-66,
-68,
-67,
-67,
-69,
-68,
-69,
-71,
-71,
-68,
-70,
-71,
-72,
-86,
-93,
-104,
-101,
-68,
-12,
19,
160,
193,
138,
-2,
-54,
-90,
-78,
-76,
-76,
-72,
-72,
-75,
-73,
-74,
-75,
-77,
-76,
-77,
-79,
-78,
-76,
-79,
-76,
-75,
-77,
-74,
-72,
-74,
-74,
-73,
-78,
-78,
-74,
-74,
-75,
-77,
-74,
-77,
-76,
-77,
-75,
-76,
-71,
-73,
-74,
-74,
-73,
-76,
-75,
-75,
-76,
-80,
-77,
-76,
-80,
-79,
-78,
-80,
-77,
-76,
-77,
-80,
-76,
-74,
-76,
-68,
-65,
-64,
-63,
-60,
-58,
-58,
-54,
-53,
-55,
-55,
-53,
-54,
-57,
-55,
-56,
-57,
-55,
-55,
-57,
-58,
-55,
-57,
-58,
-57,
-59,
-62,
-62,
-60,
-63,
-62,
-62,
-64,
-65,
-64,
-64,
-66,
-65,
-64,
-63,
-62,
-60,
-65,
-67,
-66,
-66,
-67,
-67,
-68,
-69,
-66,
-68,
-68,
-68,
-66,
-66,
-68,
-63,
-62,
-64,
-62,
-62,
-61,
-61,
-58,
-57,
-55,
-50,
-50,
-50,
-47,
-44,
-45,
-46,
-45,
-45,
-49,
-46,
-48,
-50,
-41,
-43,
-55,
-58,
-60,
-66,
-69,
-68,
-69,
-66,
-69,
-69,
-71,
-71,
-69,
-70,
-72,
-71,
-69,
-72,
-75,
-80,
-94,
-104,
-109,
-94,
-50,
7,
94,
131,
166,
74,
-48,
-94,
-93,
-76,
-78,
-74,
-74,
-72,
-74,
-72,
-75,
-74,
-74,
-76,
-79,
-76,
-75,
-79,
-75,
-74,
-75,
-77,
-75,
-77,
-78,
-75,
-75,
-76,
-74,
-76,
-77,
-78,
-76,
-78,
-79,
-75,
-74,
-75,
-75,
-73,
-73,
-76,
-73,
-73,
-78,
-77,
-78,
-78,
-77,
-77,
-79,
-81,
-79,
-81,
-80,
-76,
-77,
-78,
-73,
-70,
-68,
-66,
-64,
-60,
-62,
-55,
-55,
-55,
-56,
-55,
-59,
-61,
-60,
-58,
-61,
-60,
-60,
-61,
-62,
-62,
-63,
-61,
-58,
-59,
-63,
-63,
-64,
-66,
-63,
-64,
-67,
-68,
-66,
-65,
-65,
-64,
-64,
-66,
-64,
-63,
-65,
-68,
-66,
-66,
-69,
-65,
-66,
-69,
-67,
-64,
-63,
-63,
-66,
-64,
-66,
-63,
-61,
-62,
-58,
-55,
-54,
-53,
-51,
-49,
-47,
-43,
-44,
-46,
-48,
-46,
-50,
-50,
-51,
-44,
-50,
-55,
-63,
-64,
-70,
-72,
-71,
-72,
-72,
-70,
-74,
-70,
-68,
-67,
-71,
-69,
-72,
-74,
-73,
-81,
-92,
-93,
-101,
-111,
-96,
-53,
10,
100,
136,
152,
32,
-68,
-101,
-97,
-82,
-82,
-83,
-83,
-79,
-80,
-82,
-79,
-80,
-80,
-78,
-78,
-82,
-82,
-80,
-82,
-85,
-81,
-85,
-83,
-82,
-81,
-81,
-82,
-80,
-80,
-82,
-81,
-81,
-82,
-81,
-79,
-83,
-82,
-80,
-82,
-82,
-83,
-82,
-84,
-86,
-86,
-87,
-90,
-90,
-93,
-95,
-91,
-90,
-94,
-90,
-86,
-86,
-82,
-77,
-73,
-71,
-67,
-65,
-67,
-63,
-62,
-63,
-63,
-59,
-60,
-66,
-63,
-64,
-66,
-67,
-63,
-65,
-65,
-62,
-64,
-66,
-66,
-68,
-68,
-66,
-65,
-66,
-68,
-67,
-70,
-71,
-69,
-72,
-71,
-71,
-71,
-72,
-72,
-71,
-69,
-70,
-67,
-68,
-69,
-69,
-68,
-70,
-70,
-69,
-69,
-70,
-67,
-64,
-61,
-58,
-55,
-55,
-49,
-48,
-47,
-49,
-48,
-51,
-51,
-53,
-52,
-49,
-46,
-50,
-60,
-67,
-67,
-69,
-70,
-72,
-73,
-71,
-72,
-72,
-74,
-74,
-72,
-72,
-76,
-77,
-74,
-77,
-80,
-86,
-100,
-115,
-93,
-42,
-17,
124,
181,
146,
15,
-40,
-97,
-83,
-77,
-78,
-77,
-78,
-77,
-79,
-78,
-79,
-77,
-81,
-79,
-78,
-78,
-77,
-77,
-77,
-77,
-74,
-75,
-78,
-77,
-76,
-78,
-76,
-77,
-80,
-77,
-77,
-78,
-79,
-76,
-76,
-76,
-77,
-71,
-75,
-75,
-77,
-78,
-80,
-77,
-78,
-81,
-81,
-81,
-85,
-84,
-81,
-82,
-81,
-76,
-75,
-72,
-68,
-63,
-62,
-59,
-58,
-57,
-57,
-54,
-54,
-55,
-52,
-50,
-51,
-52,
-52,
-51,
-54,
-52,
-57,
-57,
-57,
-54,
-56,
-59,
-56,
-56,
-62,
-56,
-57,
-58,
-60,
-59,
-63,
-62,
-63,
-63,
-64,
-61,
-61,
-63,
-61,
-60,
-63,
-61,
-59,
-61,
-63,
-61,
-60,
-62,
-63,
-60,
-62,
-63,
-61,
-60,
-59,
-51,
-48,
-49,
-48,
-45,
-46,
-45,
-41,
-41,
-42,
-41,
-42,
-45,
-45,
-44,
-40,
-38,
-41,
-53,
-60,
-64,
-65,
-67,
-68,
-66,
-69,
-70,
-67,
-67,
-69,
-64,
-65,
-67,
-77,
-89,
-104,
-115,
-112,
-51,
-11,
47,
115,
140,
202,
195,
78,
-56,
-84,
-73,
-72,
-71,
-72,
-73,
-76,
-74,
-77,
-77,
-75,
-76,
-79,
-76,
-75,
-77,
-75,
-72,
-73,
-74,
-73,
-73,
-76,
-73,
-73,
-76,
-76,
-73,
-75,
-77,
-75,
-73,
-75,
-73,
-73,
-72,
-77,
-72,
-75,
-74,
-74,
-76,
-78,
-75,
-75,
-77,
-75,
-74,
-75,
-76,
-77,
-74,
-75,
-69,
-65,
-67,
-60,
-59,
-60,
-59,
-56,
-54,
-56,
-51,
-52,
-53,
-56,
-52,
-57,
-58,
-55,
-54,
-60,
-57,
-55,
-59,
-56,
-55,
-59,
-62,
-63,
-65,
-66,
-64,
-64,
-66,
-66,
-62,
-65,
-66,
-63,
-65,
-68,
-66,
-64,
-66,
-65,
-65,
-66,
-65,
-63,
-63,
-62,
-62,
-60,
-62,
-58,
-57,
-55,
-54,
-51,
-49,
-49,
-43,
-38,
-39,
-40,
-41,
-43,
-42,
-42,
-43,
-43,
-47,
-45,
-44,
-45,
-51,
-58,
-63,
-66,
-67,
-68,
-64,
-65,
-67,
-67,
-66,
-70,
-70,
-69,
-71,
-79,
-89,
-97,
-103,
-107,
-62,
-4,
89,
132,
157,
46,
-62,
-97,
-96,
-80,
-80,
-79,
-75,
-73,
-78,
-80,
-78,
-76,
-79,
-76,
-76,
-79,
-80,
-79,
-78,
-78,
-80,
-77,
-78,
-77,
-76,
-76,
-74,
-75,
-74,
-77,
-75,
-76,
-80,
-78,
-76,
-79,
-81,
-79,
-76,
-77,
-75,
-77,
-75,
-76,
-78,
-81,
-82,
-83,
-82,
-83,
-83,
-83,
-85,
-85,
-82,
-82,
-77,
-75,
-70,
-69,
-64,
-62,
-65,
-63,
-62,
-61,
-64,
-62,
-61,
-61,
-59,
-57,
-57,
-60,
-58,
-59,
-61,
-60,
-63,
-64,
-64,
-61,
-65,
-63,
-62,
-62,
-63,
-63,
-62,
-65,
-66,
-65,
-66,
-67,
-65,
-70,
-70,
-68,
-66,
-70,
-67,
-64,
-66,
-69,
-69,
-66,
-68,
-64,
-69,
-69,
-64,
-61,
-59,
-55,
-52,
-50,
-49,
-45,
-45,
-50,
-49,
-46,
-49,
-47,
-49,
-53,
-53,
-51,
-47,
-55,
-59,
-65,
-71,
-72,
-71,
-72,
-73,
-71,
-73,
-76,
-70,
-69,
-72,
-73,
-70,
-74,
-78,
-81,
-93,
-107,
-109,
-79,
-59,
60,
149,
177,
94,
32,
-94,
-98,
-84,
-82,
-83,
-80,
-79,
-81,
-80,
-81,
-81,
-84,
-83,
-82,
-86,
-86,
-83,
-84,
-84,
-85,
-86,
-88,
-83,
-82,
-82,
-86,
-85,
-84,
-85,
-85,
-85,
-85,
-83,
-83,
-84,
-85,
-83,
-84,
-84,
-84,
-84,
-87,
-84,
-86,
-88,
-90,
-89,
-93,
-98,
-96,
-96,
-98,
-94,
-93,
-92,
-91,
-87,
-86,
-82,
-78,
-75,
-74,
-69,
-66,
-69,
-67,
-65,
-66,
-65,
-64,
-64,
-66,
-63,
-64,
-64,
-65,
-66,
-69,
-70,
-66,
-67,
-71,
-69,
-72,
-70,
-70,
-70,
-71,
-71,
-72,
-72,
-74,
-75,
-76,
-76,
-76,
-74,
-75,
-75,
-75,
-74,
-76,
-73,
-72,
-73,
-76,
-72,
-75,
-75,
-72,
-75,
-75,
-72,
-70,
-72,
-70,
-66,
-64,
-64,
-60,
-60,
-58,
-54,
-51,
-53,
-54,
-55,
-57,
-59,
-57,
-59,
-58,
-50,
-55,
-60,
-69,
-74,
-79,
-80,
-77,
-80,
-82,
-78,
-79,
-83,
-80,
-78,
-81,
-83,
-80,
-94,
-108,
-116,
-129,
-128,
-72,
-38,
11,
92,
133,
184,
78,
-59,
-107,
-102,
-88,
-88,
-88,
-91,
-91,
-91,
-94,
-90,
-90,
-89,
-89,
-89,
-90,
-90,
-91,
-91,
-91,
-89,
-87,
-87,
-87,
-89,
-90,
-92,
-88,
-90,
-92,
-88,
-88,
-90,
-90,
-88,
-92,
-91,
-90,
-88,
-90,
-88,
-88,
-89,
-89,
-88,
-89,
-90,
-89,
-93,
-95,
-93,
-91,
-93,
-92,
-91,
-90,
-89,
-87,
-84,
-84,
-77,
-74,
-75,
-71,
-69,
-69,
-71,
-68,
-69,
-69,
-65,
-65,
-68,
-68,
-68,
-71,
-70,
-69,
-72,
-72,
-72,
-71,
-71,
-75,
-71,
-75,
-77,
-73,
-77,
-78,
-76,
-78,
-78,
-78,
-78,
-82,
-83,
-81,
-82,
-83,
-83,
-80,
-81,
-81,
-78,
-81,
-82,
-81,
-81,
-82,
-83,
-79,
-81,
-81,
-79,
-79,
-78,
-74,
-71,
-71,
-68,
-65,
-68,
-66,
-60,
-62,
-60,
-63,
-63,
-65,
-60,
-62,
-65,
-57,
-63,
-74,
-78,
-79,
-83,
-87,
-84,
-82,
-86,
-86,
-84,
-88,
-86,
-86,
-86,
-89,
-89,
-88,
-91,
-107,
-114,
-129,
-122,
-102,
-12,
84,
158,
150,
110,
-94,
-114,
-103,
-94,
-95,
-95,
-97,
-94,
-95,
-96,
-96,
-94,
-98,
-99,
-99,
-98,
-99,
-97,
-95,
-97,
-97,
-96,
-99,
-99,
-98,
-95,
-97,
-95,
-95,
-99,
-99,
-95,
-99,
-99,
-98,
-97,
-100,
-97,
-96,
-98,
-96,
-95,
-96,
-94,
-93,
-95,
-97,
-96,
-96,
-99,
-98,
-97,
-99,
-101,
-98,
-99,
-98,
-92,
-90,
-90,
-82,
-77,
-79,
-77,
-75,
-76,
-78,
-76,
-74,
-78,
-77,
-75,
-76,
-74,
-76,
-74,
-77,
-73,
-74,
-75,
-78,
-77,
-81,
-83,
-82,
-81,
-83,
-82,
-84,
-85,
-86,
-82,
-83,
-85,
-82,
-84,
-89,
-87,
-86,
-88,
-89,
-84,
-87,
-89,
-87,
-88,
-90,
-87,
-84,
-86,
-85,
-82,
-84,
-85,
-84,
-84,
-88,
-85,
-84,
-83,
-77,
-73,
-74,
-71,
-69,
-64,
-66,
-63,
-63,
-65,
-68,
-66,
-73,
-72,
-69,
-62,
-70,
-74,
-77,
-81,
-86,
-86,
-90,
-92,
-92,
-90,
-93,
-94,
-95,
-96,
-94,
-94,
-97,
-99,
-98,
-109,
-121,
-126,
-135,
-131,
-68,
4,
95,
145,
143,
-23,
-107,
-121,
-109,
-107,
-106,
-102,
-101,
-102,
-98,
-99,
-103,
-103,
-101,
-101,
-101,
-102,
-105,
-105,
-107,
-105,
-105,
-104,
-103,
-102,
-105,
-102,
-105,
-105,
-102,
-101,
-104,
-103,
-100,
-102,
-102,
-103,
-101,
-103,
-105,
-99,
-100,
-97,
-98,
-99,
-101,
-100,
-101,
-101,
-101,
-104,
-107,
-107,
-108,
-108,
-109,
-109,
-109,
-110,
-106,
-101,
-101,
-96,
-94,
-92,
-88,
-86,
-86,
-88,
-87,
-82,
-83,
-80,
-78,
-81,
-79,
-80,
-83,
-83,
-82,
-82,
-84,
-79,
-80,
-83,
-83,
-80,
-84,
-83,
-82,
-84,
-88,
-86,
-86,
-86,
-88,
-85,
-85,
-91,
-90,
-87,
-88,
-86,
-83,
-85,
-91,
-88,
-88,
-90,
-89,
-89,
-91,
-87,
-85,
-89,
-86,
-83,
-88,
-87,
-86,
-85,
-84,
-80,
-73,
-74,
-75,
-71,
-73,
-74,
-71,
-65,
-65,
-64,
-64,
-68,
-69,
-70,
-75,
-73,
-68,
-74,
-80,
-84,
-88,
-90,
-90,
-88,
-90,
-93,
-92,
-92,
-95,
-91,
-94,
-94,
-95,
-94,
-96,
-97,
-99,
-112,
-123,
-131,
-103,
-83,
23,
115,
162,
120,
69,
-98,
-123,
-115,
-104,
-102,
-101,
-99,
-101,
-101,
-99,
-100,
-103,
-99,
-100,
-102,
-101,
-98,
-102,
-103,
-101,
-102,
-100,
-98,
-98,
-97,
-99,
-98,
-101,
-102,
-100,
-99,
-102,
-99,
-99,
-103,
-100,
-98,
-99,
-99,
-97,
-98,
-103,
-102,
-103,
-104,
-105,
-106,
-109,
-111,
-108,
-109,
-109,
-106,
-105,
-105,
-100,
-95,
-94,
-91,
-88,
-83,
-80,
-78,
-76,
-78,
-77,
-74,
-77,
-76,
-74,
-75,
-80,
-78,
-79,
-80,
-81,
-78,
-79,
-81,
-79,
-81,
-82,
-79,
-79,
-79,
-79,
-79,
-83,
-83,
-82,
-83,
-87,
-86,
-82,
-85,
-85,
-83,
-86,
-85,
-82,
-83,
-83,
-80,
-78,
-79,
-80,
-79,
-82,
-84,
-81,
-81,
-83,
-81,
-78,
-80,
-76,
-71,
-70,
-66,
-64,
-63,
-63,
-57,
-56,
-58,
-61,
-61,
-63,
-61,
-61,
-63,
-58,
-51,
-60,
-65,
-70,
-73,
-78,
-80,
-81,
-82,
-85,
-83,
-82,
-85,
-83,
-81,
-84,
-85,
-83,
-96,
-109,
-116,
-119,
-111,
-23,
66,
155,
202,
209,
86,
-54,
-110,
-106,
-100,
-90,
-89,
-91,
-93,
-93,
-95,
-97,
-94,
-92,
-92,
-95,
-91,
-92,
-92,
-91,
-90,
-94,
-92,
-90,
-91,
-90,
-87,
-90,
-92,
-89,
-89,
-92,
-90,
-89,
-88,
-89,
-88,
-90,
-91,
-92,
-91,
-92,
-87,
-87,
-90,
-93,
-91,
-92,
-94,
-92,
-91,
-94,
-92,
-90,
-89,
-86,
-84,
-79,
-77,
-77,
-72,
-72,
-70,
-67,
-69,
-65,
-64,
-65,
-66,
-65,
-69,
-72,
-69,
-70,
-70,
-70,
-66,
-69,
-71,
-68,
-71,
-74,
-72,
-72,
-73,
-74,
-73,
-76,
-77,
-75,
-74,
-77,
-72,
-72,
-74,
-75,
-74,
-79,
-77,
-78,
-77,
-79,
-77,
-75,
-76,
-76,
-74,
-74,
-71,
-72,
-69,
-69,
-62,
-59,
-61,
-58,
-57,
-56,
-55,
-52,
-51,
-55,
-55,
-53,
-53,
-52,
-47,
-50,
-62,
-61,
-69,
-74,
-73,
-74,
-74,
-77,
-73,
-75,
-75,
-75,
-78,
-80,
-78,
-78,
-79,
-86,
-98,
-108,
-121,
-119,
-52,
19,
118,
171,
167,
-16,
-96,
-100,
-87,
-83,
-82,
-83,
-82,
-83,
-84,
-82,
-81,
-84,
-84,
-85,
-83,
-87,
-85,
-84,
-85,
-85,
-82,
-84,
-83,
-81,
-81,
-83,
-81,
-83,
-82,
-84,
-83,
-84,
-85,
-82,
-80,
-83,
-81,
-78,
-80,
-80,
-79,
-82,
-86,
-84,
-84,
-87,
-85,
-85,
-85,
-84,
-84,
-86,
-82,
-79,
-75,
-77,
-72,
-69,
-72,
-69,
-65,
-66,
-67,
-65,
-61,
-67,
-64,
-62,
-62,
-61,
-59,
-64,
-66,
-63,
-66,
-67,
-66,
-68,
-71,
-71,
-68,
-70,
-70,
-70,
-70,
-73,
-72,
-72,
-74,
-76,
-75,
-77,
-76,
-76,
-74,
-74,
-72,
-72,
-76,
-79,
-76,
-78,
-74,
-75,
-72,
-76,
-74,
-76,
-78,
-73,
-71,
-75,
-76,
-75,
-71,
-70,
-65,
-62,
-65,
-58,
-52,
-54,
-53,
-51,
-54,
-57,
-55,
-59,
-60,
-54,
-60,
-70,
-74,
-78,
-81,
-82,
-79,
-81,
-80,
-79,
-79,
-84,
-86,
-83,
-81,
-83,
-82,
-86,
-91,
-103,
-110,
-126,
-126,
-114,
-52,
25,
118,
156,
144,
-37,
-99,
-107,
-95,
-90,
-89,
-91,
-90,
-88,
-89,
-90,
-87,
-89,
-89,
-89,
-91,
-92,
-92,
-90,
-94,
-93,
-91,
-91,
-92,
-91,
-88,
-89,
-88,
-89,
-90,
-88,
-88,
-92,
-92,
-90,
-88,
-91,
-87,
-90,
-90,
-89,
-86,
-89,
-93,
-89,
-92,
-97,
-96,
-97,
-98,
-98,
-98,
-100,
-99,
-99,
-94,
-94,
-85,
-81,
-79,
-75,
-72,
-74,
-76,
-74,
-73,
-73,
-67,
-65,
-68,
-70,
-70,
-71,
-74,
-73,
-74,
-74,
-72,
-71,
-71,
-76,
-73,
-75,
-77,
-77,
-79,
-81,
-77,
-76,
-78,
-80,
-78,
-78,
-80,
-78,
-78,
-79,
-79,
-78,
-80,
-77,
-74,
-78,
-78,
-78,
-76,
-77,
-76,
-76,
-78,
-80,
-78,
-81,
-77,
-74,
-65,
-68,
-65,
-64,
-62,
-59,
-56,
-59,
-59,
-58,
-59,
-63,
-61,
-55,
-56,
-67,
-71,
-81,
-83,
-85,
-83,
-85,
-85,
-82,
-85,
-88,
-83,
-85,
-85,
-85,
-85,
-88,
-95,
-104,
-107,
-122,
-117,
-80,
-27,
5,
139,
170,
108,
-29,
-78,
-107,
-94,
-91,
-93,
-90,
-90,
-90,
-89,
-88,
-90,
-92,
-91,
-95,
-95,
-93,
-92,
-92,
-92,
-91,
-92,
-93,
-92,
-92,
-91,
-90,
-91,
-95,
-93,
-92,
-94,
-94,
-91,
-92,
-94,
-91,
-89,
-91,
-87,
-87,
-88,
-91,
-90,
-92,
-94,
-93,
-94,
-97,
-96,
-97,
-98,
-98,
-98,
-97,
-100,
-102,
-98,
-96,
-92,
-87,
-85,
-80,
-76,
-74,
-74,
-70,
-70,
-69,
-64,
-63,
-65,
-65,
-63,
-67,
-69,
-66,
-68,
-71,
-69,
-69,
-70,
-71,
-70,
-71,
-73,
-69,
-69,
-71,
-70,
-68,
-73,
-74,
-73,
-74,
-79,
-77,
-75,
-77,
-75,
-74,
-74,
-72,
-73,
-74,
-75,
-73,
-74,
-78,
-73,
-73,
-75,
-77,
-74,
-78,
-77,
-75,
-76,
-79,
-74,
-72,
-75,
-72,
-68,
-68,
-65,
-62,
-61,
-61,
-56,
-53,
-54,
-55,
-53,
-56,
-58,
-58,
-56,
-60,
-52,
-49,
-52,
-62,
-68,
-74,
-76,
-78,
-74,
-77,
-77,
-76,
-78,
-79,
-77,
-80,
-81,
-80,
-80,
-84,
-88,
-100,
-104,
-112,
-121,
-128,
-97,
-79,
15,
110,
180,
172,
132,
-73,
-104,
-96,
-89,
-86,
-89,
-91,
-90,
-89,
-91,
-91,
-90,
-91,
-91,
-88,
-87,
-92,
-87,
-91,
-90,
-91,
-89,
-91,
-91,
-89,
-88,
-90,
-85,
-86,
-89,
-86,
-85,
-88,
-90,
-90,
-89,
-90,
-88,
-90,
-91,
-88,
-84,
-87,
-89,
-85,
-84,
-88,
-86,
-87,
-89,
-90,
-87,
-90,
-91,
-89,
-90,
-90,
-88,
-83,
-85,
-80,
-75,
-73,
-74,
-70,
-68,
-70,
-68,
-68,
-67,
-68,
-64,
-68,
-66,
-65,
-65,
-69,
-65,
-65,
-67,
-66,
-65,
-67,
-71,
-68,
-70,
-72,
-72,
-71,
-73,
-73,
-71,
-72,
-74,
-74,
-73,
-77,
-77,
-76,
-78,
-78,
-77,
-78,
-81,
-78,
-77,
-80,
-78,
-76,
-76,
-74,
-74,
-75,
-77,
-77,
-75,
-78,
-74,
-74,
-76,
-76,
-74,
-72,
-70,
-64,
-60,
-62,
-56,
-51,
-55,
-54,
-55,
-60,
-61,
-59,
-58,
-63,
-62,
-58,
-59,
-53,
-59,
-67,
-73,
-75,
-77,
-81,
-77,
-79,
-80,
-82,
-80,
-83,
-84,
-84,
-82,
-86,
-88,
-100,
-108,
-124,
-119,
-80,
-31,
0,
136,
169,
128,
13,
-38,
-115,
-113,
-99,
-92,
-90,
-88,
-90,
-90,
-87,
-90,
-92,
-88,
-90,
-92,
-91,
-92,
-92,
-89,
-86,
-86,
-88,
-86,
-87,
-90,
-88,
-87,
-88,
-85,
-83,
-85,
-88,
-85,
-90,
-88,
-90,
-88,
-90,
-87,
-86,
-90,
-87,
-86,
-87,
-89,
-86,
-88,
-87,
-86,
-84,
-85,
-88,
-87,
-89,
-91,
-87,
-86,
-87,
-86,
-84,
-86,
-79,
-74,
-73,
-71,
-73,
-68,
-69,
-71,
-67,
-67,
-69,
-66,
-71,
-72,
-71,
-68,
-70,
-68,
-68,
-70,
-72,
-68,
-74,
-75,
-73,
-76,
-79,
-76,
-75,
-76,
-76,
-76,
-79,
-82,
-78,
-77,
-83,
-80,
-77,
-80,
-84,
-82,
-86,
-86,
-84,
-83,
-84,
-82,
-78,
-80,
-82,
-80,
-82,
-84,
-82,
-83,
-86,
-81,
-76,
-75,
-70,
-67,
-67,
-62,
-60,
-59,
-60,
-59,
-61,
-63,
-65,
-64,
-68,
-68,
-64,
-64,
-75,
-77,
-81,
-84,
-85,
-81,
-84,
-88,
-90,
-88,
-90,
-89,
-88,
-90,
-89,
-87,
-89,
-98,
-102,
-117,
-129,
-102,
-55,
-27,
120,
159,
91,
-39,
-80,
-111,
-97,
-92,
-91,
-90,
-92,
-94,
-95,
-97,
-94,
-92,
-95,
-95,
-91,
-95,
-91,
-88,
-90,
-91,
-91,
-95,
-93,
-91,
-92,
-95,
-94,
-92,
-90,
-90,
-87,
-88,
-91,
-90,
-87,
-91,
-89,
-88,
-89,
-91,
-90,
-90,
-95,
-94,
-94,
-93,
-94,
-92,
-96,
-100,
-99,
-96,
-100,
-97,
-95,
-93,
-90,
-84,
-82,
-84,
-80,
-72,
-74,
-68,
-65,
-68,
-72,
-71,
-73,
-70,
-66,
-70,
-72,
-72,
-67,
-67,
-70,
-68,
-72,
-72,
-70,
-72,
-76,
-74,
-73,
-73,
-76,
-74,
-79,
-80,
-78,
-74,
-78,
-75,
-75,
-79,
-80,
-79,
-80,
-81,
-81,
-78,
-81,
-79,
-75,
-79,
-78,
-76,
-76,
-76,
-74,
-75,
-78,
-75,
-69,
-71,
-67,
-63,
-61,
-63,
-60,
-56,
-58,
-54,
-55,
-55,
-58,
-57,
-58,
-60,
-59,
-54,
-58,
-63,
-67,
-72,
-82,
-79,
-79,
-83,
-80,
-80,
-81,
-81,
-82,
-84,
-85,
-82,
-84,
-90,
-94,
-108,
-122,
-110,
-63,
-39,
93,
165,
160,
49,
-16,
-108,
-101,
-90,
-89,
-88,
-87,
-84,
-88,
-89,
-86,
-87,
-92,
-92,
-88,
-90,
-91,
-87,
-89,
-92,
-90,
-87,
-88,
-85,
-86,
-87,
-89,
-89,
-89,
-89,
-90,
-88,
-90,
-87,
-85,
-86,
-86,
-85,
-88,
-88,
-86,
-86,
-89,
-87,
-88,
-91,
-92,
-92,
-95,
-97,
-95,
-95,
-96,
-94,
-93,
-94,
-91,
-88,
-87,
-84,
-80,
-73,
-75,
-71,
-69,
-66,
-65,
-65,
-64,
-66,
-64,
-62,
-65,
-63,
-64,
-64,
-67,
-66,
-67,
-67,
-65,
-67,
-69,
-68,
-65,
-68,
-66,
-63,
-67,
-68,
-66,
-66,
-71,
-68,
-68,
-68,
-72,
-70,
-69,
-74,
-69,
-71,
-72,
-70,
-70,
-70,
-67,
-67,
-69,
-70,
-68,
-69,
-72,
-71,
-71,
-75,
-72,
-68,
-71,
-72,
-69,
-68,
-68,
-63,
-56,
-57,
-56,
-54,
-50,
-49,
-46,
-48,
-51,
-50,
-47,
-50,
-50,
-43,
-51,
-59,
-63,
-66,
-71,
-69,
-73,
-71,
-72,
-72,
-73,
-76,
-74,
-72,
-75,
-74,
-73,
-73,
-85,
-97,
-109,
-119,
-112,
-38,
27,
127,
199,
210,
72,
-60,
-103,
-86,
-78,
-81,
-84,
-81,
-83,
-84,
-86,
-84,
-86,
-86,
-85,
-83,
-86,
-86,
-86,
-88,
-87,
-84,
-85,
-86,
-87,
-85,
-86,
-84,
-84,
-84,
-87,
-86,
-88,
-87,
-87,
-86,
-88,
-87,
-85,
-86,
-85,
-84,
-84,
-86,
-86,
-86,
-90,
-88,
-88,
-87,
-88,
-85,
-85,
-86,
-82,
-79,
-77,
-74,
-68,
-70,
-65,
-61,
-63,
-65,
-64,
-64,
-64,
-64,
-63,
-64,
-65,
-63,
-64,
-64,
-66,
-67,
-70,
-67,
-68,
-67,
-70,
-68,
-72,
-72,
-71,
-71,
-72,
-71,
-71,
-72,
-74,
-72,
-76,
-77,
-74,
-72,
-75,
-74,
-71,
-73,
-73,
-71,
-72,
-71,
-71,
-70,
-73,
-70,
-70,
-71,
-71,
-69,
-70,
-69,
-65,
-59,
-59,
-55,
-50,
-52,
-49,
-45,
-50,
-53,
-53,
-51,
-56,
-54,
-45,
-46,
-59,
-58,
-66,
-69,
-67,
-65,
-70,
-72,
-72,
-73,
-76,
-74,
-76,
-79,
-77,
-77,
-77,
-76,
-76,
-80,
-98,
-105,
-120,
-102,
-78,
36,
130,
163,
73,
4,
-106,
-87,
-80,
-83,
-82,
-79,
-85,
-83,
-81,
-85,
-84,
-83,
-84,
-85,
-82,
-79,
-82,
-81,
-77,
-81,
-80,
-78,
-78,
-81,
-79,
-80,
-84,
-80,
-80,
-82,
-80,
-78,
-81,
-82,
-81,
-78,
-80,
-77,
-79,
-82,
-82,
-79,
-84,
-86,
-84,
-86,
-85,
-85,
-83,
-85,
-82,
-79,
-82,
-82,
-79,
-77,
-75,
-71,
-69,
-70,
-67,
-65,
-69,
-69,
-66,
-65,
-69,
-65,
-65,
-68,
-70,
-72,
-76,
-75,
-72,
-74,
-75,
-74,
-73,
-74,
-74,
-72,
-75,
-76,
-76,
-79,
-80,
-76,
-76,
-77,
-80,
-80,
-82,
-81,
-80,
-78,
-82,
-80,
-78,
-79,
-80,
-80,
-81,
-82,
-83,
-77,
-79,
-75,
-76,
-78,
-76,
-68,
-69,
-69,
-65,
-66,
-68,
-62,
-60,
-62,
-59,
-59,
-61,
-63,
-61,
-64,
-62,
-57,
-66,
-69,
-79,
-82,
-87,
-89,
-88,
-86,
-88,
-85,
-85,
-88,
-88,
-86,
-86,
-90,
-89,
-99,
-112,
-123,
-133,
-128,
-78,
-46,
-2,
68,
103,
167,
96,
-31,
-106,
-118,
-100,
-95,
-97,
-96,
-95,
-95,
-100,
-100,
-95,
-97,
-98,
-95,
-99,
-99,
-96,
-92,
-94,
-96,
-98,
-98,
-98,
-96,
-97,
-99,
-96,
-95,
-97,
-96,
-97,
-96,
-97,
-98,
-101,
-103,
-99,
-94,
-98,
-100,
-100,
-102,
-101,
-101,
-106,
-108,
-106,
-109,
-111,
-108,
-106,
-108,
-107,
-104,
-100,
-97,
-93,
-88,
-87,
-82,
-81,
-82,
-78,
-77,
-79,
-80,
-77,
-76,
-80,
-75,
-73,
-77,
-80,
-76,
-78,
-79,
-78,
-78,
-81,
-81,
-81,
-81,
-82,
-80,
-85,
-85,
-85,
-84,
-86,
-83,
-83,
-85,
-85,
-83,
-85,
-87,
-85,
-86,
-89,
-87,
-85,
-88,
-87,
-84,
-87,
-87,
-85,
-86,
-86,
-84,
-85,
-85,
-85,
-84,
-85,
-86,
-83,
-76,
-76,
-74,
-67,
-67,
-65,
-60,
-62,
-65,
-65,
-67,
-69,
-68,
-67,
-65,
-67,
-70,
-78,
-84,
-83,
-85,
-87,
-87,
-86,
-87,
-91,
-87,
-89,
-91,
-90,
-89,
-94,
-91,
-90,
-89,
-97,
-106,
-116,
-128,
-128,
-69,
-6,
88,
159,
170,
31,
-79,
-114,
-109,
-101,
-96,
-97,
-96,
-95,
-98,
-100,
-98,
-100,
-99,
-99,
-97,
-99,
-96,
-96,
-98,
-97,
-96,
-98,
-97,
-96,
-97,
-99,
-97,
-95,
-98,
-99,
-96,
-99,
-97,
-97,
-96,
-99,
-98,
-97,
-99,
-98,
-95,
-97,
-96,
-100,
-97,
-100,
-100,
-100,
-104,
-104,
-102,
-104,
-105,
-102,
-101,
-105,
-101,
-98,
-99,
-94,
-88,
-86,
-81,
-82,
-80,
-79,
-77,
-75,
-76,
-74,
-73,
-75,
-75,
-74,
-73,
-77,
-73,
-74,
-78,
-77,
-75,
-78,
-79,
-78,
-78,
-81,
-79,
-80,
-81,
-79,
-76,
-77,
-79,
-79,
-79,
-81,
-80,
-79,
-80,
-82,
-79,
-81,
-83,
-81,
-80,
-83,
-79,
-77,
-79,
-81,
-77,
-79,
-82,
-79,
-81,
-83,
-81,
-79,
-80,
-80,
-78,
-77,
-73,
-69,
-63,
-68,
-63,
-62,
-64,
-60,
-58,
-58,
-61,
-60,
-64,
-66,
-64,
-63,
-63,
-55,
-60,
-70,
-73,
-75,
-83,
-86,
-83,
-81,
-83,
-85,
-83,
-85,
-86,
-83,
-84,
-87,
-87,
-89,
-93,
-110,
-118,
-128,
-114,
-94,
-8,
80,
161,
183,
167,
-25,
-97,
-112,
-103
};
