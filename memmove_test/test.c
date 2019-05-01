/*
 * Copyright (C) 2018 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <string.h>

int main()
{
  printf("Hello !\n");

  int buffer[2] = {3,4};

  printf("before buffer 0 = %d, buffer 1 = %d\n", buffer[0], buffer[1]);

  memcpy(&buffer[1], buffer, sizeof(int));

  printf("after buffer 0 = %d, buffer 1 = %d\n", buffer[0], buffer[1]);

  return 0;
}
